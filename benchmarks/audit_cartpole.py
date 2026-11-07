import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json

from src.numpy_dqn.agent import NumpyDQNAgent as Agent

def run_100_episodes_benchmark(model_path="weights/numpy_model.npz", num_episodes=100):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = gym.make("CartPole-v1")
    
    scores = []
    terminations = []  # "truncated" or "angle_fail" or "pos_fail"
    final_states = []
    
    all_x = []
    all_theta = []
    action_switch_counts = []
    total_actions = []
    
    # Store trajectories for plotting (first 5 episodes)
    sample_trajectories = []

    for seed in range(num_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        score = 0
        prev_action = None
        switches = 0
        actions_in_ep = 0
        
        ep_x = []
        ep_v = []
        ep_theta = []
        ep_omega = []
        ep_actions = []

        while not done:
            action = agent.choose_action(state)
            if prev_action is not None and action != prev_action:
                switches += 1
            prev_action = action
            actions_in_ep += 1
            
            x, v, theta, omega = state.squeeze()
            ep_x.append(float(x))
            ep_v.append(float(v))
            ep_theta.append(float(theta))
            ep_omega.append(float(omega))
            ep_actions.append(int(action))

            all_x.append(float(x))
            all_theta.append(float(theta))

            next_state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            
            if done:
                final_states.append({
                    "seed": seed,
                    "x": float(next_state[0]),
                    "v": float(next_state[1]),
                    "theta": float(next_state[2]),
                    "omega": float(next_state[3]),
                    "score": float(score),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated)
                })
                if truncated:
                    terminations.append("truncated (500 steps reached)")
                elif terminated:
                    # check failure cause
                    # CartPole-v1: failure if |x| > 2.4 or |theta| > 12 * 2 * pi / 360 (~0.2094 rad)
                    is_x = abs(next_state[0]) > 2.4
                    is_theta = abs(next_state[2]) > (12 * 2 * np.pi / 360)
                    if is_x and is_theta:
                        terminations.append("angle and position limit")
                    elif is_x:
                        terminations.append("position limit (|x| > 2.4)")
                    elif is_theta:
                        terminations.append("angle limit (|theta| > 12 deg)")
                    else:
                        terminations.append("early terminated (other)")
            
            state = next_state

        scores.append(score)
        action_switch_counts.append(switches)
        total_actions.append(actions_in_ep)
        
        if seed < 5:
            sample_trajectories.append({
                "seed": seed,
                "x": ep_x,
                "v": ep_v,
                "theta": ep_theta,
                "omega": ep_omega,
                "actions": ep_actions
            })

    env.close()

    scores_arr = np.array(scores)
    stats = {
        "num_episodes": num_episodes,
        "mean_score": float(np.mean(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "min_score": float(np.min(scores_arr)),
        "max_score": float(np.max(scores_arr)),
        "success_rate_500": float(np.mean(scores_arr == 500.0) * 100),
        "scores_above_475": float(np.mean(scores_arr >= 475.0) * 100),
        "solved_gymnasium_criterion": bool(np.mean(scores_arr) >= 475.0),
        "termination_breakdown": {
            k: terminations.count(k) for k in set(terminations)
        },
        "mean_cart_position": float(np.mean(np.abs(all_x))),
        "max_cart_position": float(np.max(np.abs(all_x))),
        "mean_pole_angle_deg": float(np.mean(np.abs(all_theta)) * 180 / np.pi),
        "max_pole_angle_deg": float(np.max(np.abs(all_theta)) * 180 / np.pi),
        "mean_action_switch_rate": float(np.sum(action_switch_counts) / np.sum(total_actions) * 100)
    }

    return stats, scores, final_states, sample_trajectories

def run_extended_horizon_test(model_path="best_model.npz", num_episodes=20, max_steps=2000):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = gym.make("CartPole-v1", max_episode_steps=max_steps)
    extended_scores = []
    reasons = []

    for seed in range(num_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
            if done:
                if truncated:
                    reasons.append("reached_horizon")
                elif abs(next_state[0]) > 2.4:
                    reasons.append("cart_drift_exit")
                else:
                    reasons.append("pole_fall")
            state = next_state
        extended_scores.append(steps)
    env.close()

    return {
        "max_steps": max_steps,
        "episodes": num_episodes,
        "mean_steps": float(np.mean(extended_scores)),
        "std_steps": float(np.std(extended_scores)),
        "min_steps": float(np.min(extended_scores)),
        "max_steps_achieved": float(np.max(extended_scores)),
        "reasons": {k: reasons.count(k) for k in set(reasons)},
        "all_steps": extended_scores
    }

def run_stress_initial_angles(model_path="best_model.npz"):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = gym.make("CartPole-v1")
    # Test angles from -0.18 to +0.18 radians (-10.3 deg to +10.3 deg; max fail is ~12.0 deg / 0.2094 rad)
    test_angles = np.linspace(-0.18, 0.18, 25)
    results = []

    for angle in test_angles:
        env.reset(seed=42)
        env.unwrapped.state = np.array([0.0, 0.0, angle, 0.0])
        state = env.unwrapped.state.copy()
        
        steps = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
            state = next_state
            if steps >= 500:
                break
        results.append({
            "angle_rad": float(angle),
            "angle_deg": float(angle * 180 / np.pi),
            "steps": steps,
            "recovered": bool(steps >= 500)
        })
    env.close()
    return results

def run_stress_initial_positions(model_path="best_model.npz"):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = gym.make("CartPole-v1")
    # Test cart positions from -2.0 to +2.0 (fail limit is 2.4)
    test_positions = np.linspace(-2.0, 2.0, 25)
    results = []

    for pos in test_positions:
        env.reset(seed=42)
        env.unwrapped.state = np.array([pos, 0.0, 0.0, 0.0])
        state = env.unwrapped.state.copy()
        
        steps = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
            state = next_state
            if steps >= 500:
                break
        results.append({
            "position": float(pos),
            "steps": steps,
            "recovered": bool(steps >= 500)
        })
    env.close()
    return results

def run_impulse_perturbation_test(model_path="best_model.npz"):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = gym.make("CartPole-v1")
    
    # At step 100, apply angular velocity kick delta_omega
    kicks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for kick in kicks:
        state, _ = env.reset(seed=42)
        steps = 0
        done = False
        while not done:
            if steps == 100:
                # Inject kick to angular velocity
                curr_state = env.unwrapped.state.copy()
                curr_state[3] += kick
                env.unwrapped.state = curr_state
                state = curr_state
                
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
            state = next_state
            if steps >= 500:
                break
        results.append({
            "angular_velocity_kick": float(kick),
            "survived_steps": steps,
            "withstood": bool(steps >= 500)
        })
    env.close()
    return results

def run_action_noise_stress(model_path="best_model.npz"):
    agent = Agent(input_size=4, hidden_size=64, output_size=2)
    agent.load(model_path)
    
    noise_rates = [0.0, 0.01, 0.02, 0.05, 0.10]
    results = []
    
    env = gym.make("CartPole-v1")
    for noise in noise_rates:
        agent.epsilon = noise
        scores = []
        for seed in range(20):
            state, _ = env.reset(seed=seed)
            done = False
            score = 0
            while not done:
                action = agent.choose_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                score += reward
                done = terminated or truncated
            scores.append(score)
        results.append({
            "noise_rate_epsilon": noise,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "success_rate": float(np.mean(np.array(scores) == 500.0) * 100)
        })
    env.close()
    return results

if __name__ == "__main__":
    print(">>> 1. Starting 100-episode official Gymnasium benchmark (seeds 0-99)...")
    benchmark_stats, scores, final_states, sample_trajs = run_100_episodes_benchmark()
    
    print(">>> Benchmark results:")
    for k, v in benchmark_stats.items():
        print(f"  {k}: {v}")

    print("\n>>> 2. Starting Extended Horizon evaluation (max 2000 steps, 20 seeds)...")
    ext_results = run_extended_horizon_test()
    print(">>> Extended Horizon results:", ext_results)

    print("\n>>> 3. Starting Stress Test: Initial Angles...")
    angle_results = run_stress_initial_angles()
    print(f"  Tested {len(angle_results)} angles. Recoveries: {sum(r['recovered'] for r in angle_results)}/{len(angle_results)}")

    print("\n>>> 4. Starting Stress Test: Initial Positions...")
    pos_results = run_stress_initial_positions()
    print(f"  Tested {len(pos_results)} positions. Recoveries: {sum(r['recovered'] for r in pos_results)}/{len(pos_results)}")

    print("\n>>> 5. Starting Stress Test: Angular Velocity Impulses (Kicks at step 100)...")
    impulse_results = run_impulse_perturbation_test()
    for r in impulse_results:
        print(f"  Kick: {r['angular_velocity_kick']:+.2f} rad/s -> Steps: {r['survived_steps']}")

    print("\n>>> 6. Starting Stress Test: Action Noise injection...")
    noise_results = run_action_noise_stress()
    for r in noise_results:
        print(f"  Epsilon {r['noise_rate_epsilon']:.2f} -> Mean score: {r['mean_score']:.1f} (Success: {r['success_rate']}%)")

    # Save complete audit report data to json
    full_audit_data = {
        "benchmark_stats": benchmark_stats,
        "extended_horizon": ext_results,
        "stress_initial_angles": angle_results,
        "stress_initial_positions": pos_results,
        "stress_impulse": impulse_results,
        "stress_action_noise": noise_results,
        "scores_all_100": scores,
        "final_states_failures": [s for s in final_states if s["score"] < 500.0]
    }

    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_results.json")
    with open(out_json, "w") as f:
        json.dump(full_audit_data, f, indent=2)
    print(f"\n>>> Audit data saved to {out_json}")
