import argparse
import os
import gymnasium as gym
import numpy as np
from PIL import Image

from src.numpy_dqn.agent import NumpyDQNAgent


def run_episode(agent, env, max_steps=500, capture_frames=False):
    state, _ = env.reset()
    state = state.reshape(1, -1)
    frames = []
    total_reward = 0

    for _ in range(max_steps):
        if capture_frames:
            frames.append(Image.fromarray(env.render()))

        action = agent.choose_action(state, greedy=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state.reshape(1, -1)
        total_reward += reward

        if terminated or truncated:
            break

    if capture_frames:
        frames.append(Image.fromarray(env.render()))

    return total_reward, frames


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize trained CartPole DQN agent.")
    parser.add_argument("--backend", type=str, choices=["numpy", "torch"], default="numpy", help="Backend implementation.")
    parser.add_argument("--model", type=str, default="", help="Path to model checkpoint. Defaults to weights/.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true", help="Render environment window using Pygame.")
    parser.add_argument("--record-gif", type=str, default="", help="Path to save animated GIF.")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second for the saved GIF.")
    args = parser.parse_args()

    if args.backend == "numpy":
        model_path = args.model or "weights/numpy_model.npz"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NumPy weights not found at: {model_path}")
        agent = NumpyDQNAgent(input_size=4, hidden_size=64, output_size=2)
        agent.load(model_path)
    else:
        from src.torch_dqn.agent import TorchDQNAgent

        model_path = args.model or "weights/torch_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch weights not found at: {model_path}")
        agent = TorchDQNAgent(input_size=4, hidden_size=64, output_size=2)
        agent.load(model_path)

    # Pure greedy policy for evaluation
    agent.epsilon = 0.0
    print(f"Loaded {args.backend.upper()} agent from: {model_path}")

    if args.record_gif:
        os.makedirs(os.path.dirname(args.record_gif) or ".", exist_ok=True)
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        print("Recording evaluation episode for GIF...")
        score, frames = run_episode(agent, env, capture_frames=True)
        env.close()

        if frames:
            # Downsample to 25 fps to maintain small GIF size
            sampled_frames = frames[::2]
            duration_ms = int(1000 / (args.fps / 2))
            sampled_frames[0].save(
                args.record_gif,
                save_all=True,
                append_images=sampled_frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0,
            )
            print(f"Saved GIF to {args.record_gif} ({len(sampled_frames)} frames, score: {score:.0f})")

    render_mode = "human" if args.render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    print(f"\nRunning {args.episodes} evaluation episodes (greedy policy, eps=0)...")
    scores = []
    for ep in range(args.episodes):
        score, _ = run_episode(agent, env)
        scores.append(score)
        print(f"Episode {ep + 1:2d}/{args.episodes}: {score:.0f} steps")
    env.close()

    print(
        f"\nSummary: Mean = {np.mean(scores):.1f} ± {np.std(scores):.1f} | "
        f"Min = {np.min(scores):.0f} | Max = {np.max(scores):.0f}"
    )


if __name__ == "__main__":
    main()
