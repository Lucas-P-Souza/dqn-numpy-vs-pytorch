import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import psutil

from src.common.utils import set_seed
from src.numpy_dqn.agent import NumpyDQNAgent
from src.torch_dqn.agent import TorchDQNAgent


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def train_single_agent(agent_type: str, episodes: int = 300, seed: int = 42, device: str = "cpu"):
    set_seed(seed)
    env = gym.make("CartPole-v1")

    if agent_type == "numpy":
        agent = NumpyDQNAgent(input_size=4, hidden_size=64, output_size=2)
    elif agent_type == "torch":
        agent = TorchDQNAgent(input_size=4, hidden_size=64, output_size=2, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    scores = []
    losses = []
    train_step_times = []
    total_steps = 0
    batch_size = 64
    solved_at_episode = None

    mem_before = get_process_memory_mb()
    peak_mem = mem_before
    start_time = time.perf_counter()

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        state = state.reshape(1, -1)
        done = False
        ep_score = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_fmt = next_state.reshape(1, -1)
            done = terminated or truncated

            # Terminal penalty discourages premature pole collapse
            train_reward = reward
            if terminated:
                train_reward = -50.0

            agent.remember(state, action, train_reward, next_state_fmt, done)

            t0 = time.perf_counter()
            loss = agent.train_step(batch_size=batch_size, lr=0.00025)
            t_train = time.perf_counter() - t0
            if loss > 0.0:
                train_step_times.append(t_train)
                losses.append(loss)

            state = next_state_fmt
            ep_score += 1
            total_steps += 1

            # Periodic target network sync
            if total_steps % 1000 == 0:
                agent.update_target_network()

        # Exploration epsilon decay
        if agent.epsilon > 0.01:
            agent.epsilon -= 0.001

        scores.append(ep_score)
        current_mem = get_process_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

        # Save model if threshold is reached
        if ep_score >= 475 or (len(scores) >= 10 and np.mean(scores[-10:]) > 400):
            save_name = "weights/numpy_model.npz" if agent_type == "numpy" else "weights/torch_model.pt"
            os.makedirs("weights", exist_ok=True)
            agent.save(save_name)

        # Track solved criterion: rolling average over last 50 episodes >= 475
        if len(scores) >= 50 and np.mean(scores[-50:]) >= 475.0 and solved_at_episode is None:
            solved_at_episode = ep

        if (ep + 1) % 50 == 0 or ep == episodes - 1:
            avg_10 = np.mean(scores[-10:])
            elapsed = time.perf_counter() - start_time
            fps = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"[{agent_type.upper():<5} | Ep {ep + 1:3d}/{episodes}] "
                f"Avg(10): {avg_10:5.1f} | Eps: {agent.epsilon:.2f} | "
                f"FPS: {fps:5.0f} | Mem: {peak_mem:5.1f}MB"
            )

    total_time = time.perf_counter() - start_time
    env.close()

    avg_train_step_us = (np.mean(train_step_times) * 1e6) if train_step_times else 0.0

    return {
        "agent": agent_type,
        "device": device,
        "episodes": episodes,
        "total_steps": total_steps,
        "total_time_seconds": round(total_time, 3),
        "steps_per_second": round(total_steps / total_time, 1),
        "avg_train_step_microseconds": round(avg_train_step_us, 2),
        "initial_memory_mb": round(mem_before, 2),
        "peak_memory_mb": round(peak_mem, 2),
        "memory_growth_mb": round(peak_mem - mem_before, 2),
        "final_mean_score_last_50": round(float(np.mean(scores[-50:])), 2) if len(scores) >= 50 else round(float(np.mean(scores)), 2),
        "solved_at_episode": solved_at_episode,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Run side-by-side benchmark of NumPy vs PyTorch DQN.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes per run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output", type=str, default="benchmarks/benchmark_data.json", help="Output JSON path.")
    args = parser.parse_args()

    print("=" * 65)
    print("      STARTING BENCHMARK: NUMPY (FROM SCRATCH) vs. PYTORCH")
    print("=" * 65)

    print("\n--- Running DQN in Pure NumPy (CPU) ---")
    numpy_results = train_single_agent("numpy", episodes=args.episodes, seed=args.seed)

    print("\n--- Running DQN in PyTorch (CPU) ---")
    torch_cpu_results = train_single_agent("torch", episodes=args.episodes, seed=args.seed, device="cpu")

    results = {
        "metadata": {
            "episodes": args.episodes,
            "seed": args.seed,
            "environment": "CartPole-v1",
            "batch_size": 64,
            "hidden_size": 64,
        },
        "numpy": numpy_results,
        "torch_cpu": torch_cpu_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("                 FINAL BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<32} | {'NumPy (Scratch)':<15} | {'PyTorch (CPU)':<15}")
    print("-" * 65)
    print(f"{'Throughput (Steps/second)':<32} | {numpy_results['steps_per_second']:<15} | {torch_cpu_results['steps_per_second']:<15}")
    print(f"{'Total Time (s)':<32} | {numpy_results['total_time_seconds']:<15} | {torch_cpu_results['total_time_seconds']:<15}")
    print(f"{'Optimization Step (µs/step)':<32} | {numpy_results['avg_train_step_microseconds']:<15} | {torch_cpu_results['avg_train_step_microseconds']:<15}")
    print(f"{'Peak RAM (MB)':<32} | {numpy_results['peak_memory_mb']:<15} | {torch_cpu_results['peak_memory_mb']:<15}")
    print(f"{'Mean Score (Last 50 ep)':<32} | {numpy_results['final_mean_score_last_50']:<15} | {torch_cpu_results['final_mean_score_last_50']:<15}")
    print("=" * 65)
    print(f"Raw metrics saved to: '{args.output}'")


if __name__ == "__main__":
    main()
