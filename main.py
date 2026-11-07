import argparse
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.common.utils import moving_average, set_seed
from src.numpy_dqn.agent import NumpyDQNAgent


def main():
    parser = argparse.ArgumentParser(description="Train DQN Agent on CartPole-v1.")
    parser.add_argument("--backend", type=str, choices=["numpy", "torch"], default="numpy", help="Engine backend.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-path", type=str, default="", help="Checkpoint destination.")
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make("CartPole-v1")

    if args.backend == "numpy":
        agent = NumpyDQNAgent(input_size=4, hidden_size=64, output_size=2)
        save_path = args.save_path or "weights/numpy_model.npz"
    else:
        from src.torch_dqn.agent import TorchDQNAgent

        agent = TorchDQNAgent(input_size=4, hidden_size=64, output_size=2)
        save_path = args.save_path or "weights/torch_model.pt"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f"Training DQN [{args.backend.upper()}] for {args.episodes} episodes...")

    scores = []
    best_score = 0
    total_steps = 0

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        state = state.reshape(1, -1)
        done = False
        ep_score = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_fmt = next_state.reshape(1, -1)
            done = terminated or truncated

            # Penalize terminal state to avoid premature pole collapse
            train_reward = reward
            if terminated:
                train_reward = -50.0

            agent.remember(state, action, train_reward, next_state_fmt, done)
            agent.train_step(batch_size=args.batch_size, lr=args.lr)

            state = next_state_fmt
            ep_score += 1
            total_steps += 1

            # Sync target network every 1000 global steps
            if total_steps % 1000 == 0:
                agent.update_target_network()

        # Linear exploration decay
        if agent.epsilon > 0.01:
            agent.epsilon -= 0.001

        scores.append(ep_score)

        if ep_score >= best_score:
            best_score = ep_score
            agent.save(save_path)

        if (ep + 1) % 25 == 0 or ep == args.episodes - 1:
            avg_10 = np.mean(scores[-10:])
            print(f"[Ep {ep + 1:3d}/{args.episodes}] Score: {ep_score:3d} | Avg(10): {avg_10:5.1f} | Eps: {agent.epsilon:.2f}")

    env.close()
    print(f"\nBest score recorded: {best_score} steps.")
    print(f"Model saved to: '{save_path}'")

    # Plot learning curve
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(scores, label="Episode Score", alpha=0.25, color="gray")
    if len(scores) >= 20:
        smoothed = moving_average(scores, window=20)
        plt.plot(range(len(smoothed)), smoothed, label="Rolling Mean (20 ep)", color="#1f77b4", linewidth=2)
    plt.axhline(475, color="green", linestyle="--", label="Gymnasium Solved Threshold (475)")
    plt.title(f"CartPole-v1 DQN Training ({args.backend.upper()})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_file = f"assets/train_{args.backend}.png"
    plt.savefig(plot_file)
    print(f"Training plot saved to: '{plot_file}'")


if __name__ == "__main__":
    main()
