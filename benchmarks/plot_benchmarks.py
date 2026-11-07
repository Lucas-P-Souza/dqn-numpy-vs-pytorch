import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from src.common.utils import moving_average


def generate_benchmark_plots(data_path: str = "benchmarks/benchmark_data.json", assets_dir: str = "assets"):
    os.makedirs(assets_dir, exist_ok=True)
    with open(data_path, "r") as f:
        data = json.load(f)

    numpy_res = data["numpy"]
    torch_res = data["torch_cpu"]

    # Clean publication-style layout
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Comparative Learning Curve
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    episodes = len(numpy_res["scores"])
    x = np.arange(1, episodes + 1)

    # NumPy
    ax.plot(x, numpy_res["scores"], color="#1f77b4", alpha=0.15, linewidth=1)
    np_smooth = moving_average(numpy_res["scores"], window=20)
    ax.plot(x[: len(np_smooth)], np_smooth, color="#1f77b4", linewidth=2.2, label="NumPy (From Scratch) - Rolling Mean")

    # PyTorch
    ax.plot(x, torch_res["scores"], color="#d62728", alpha=0.15, linewidth=1)
    torch_smooth = moving_average(torch_res["scores"], window=20)
    ax.plot(x[: len(torch_smooth)], torch_smooth, color="#d62728", linewidth=2.2, label="PyTorch (CPU) - Rolling Mean")

    # Gymnasium target threshold
    ax.axhline(475, color="#2ca02c", linestyle="--", linewidth=1.5, label="Gymnasium Solved Threshold (475)")

    ax.set_title("Convergence Curve: DQN NumPy vs. PyTorch (CartPole-v1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Episode Score (Balanced Steps)", fontsize=11)
    ax.set_ylim(-10, 520)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=9.5, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(assets_dir, "benchmark_learning.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    # 2. Computational Throughput & Step Latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    frameworks = ["NumPy\n(Scratch)", "PyTorch\n(CPU)"]
    fps_vals = [numpy_res["steps_per_second"], torch_res["steps_per_second"]]
    step_times = [numpy_res["avg_train_step_microseconds"], torch_res["avg_train_step_microseconds"]]
    colors = ["#1f77b4", "#d62728"]

    # Throughput (FPS)
    bars1 = ax1.bar(frameworks, fps_vals, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8)
    ax1.set_title("Execution Throughput (FPS)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Steps / Second", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + (max(fps_vals) * 0.02), f"{yval:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, max(fps_vals) * 1.15)

    # Optimization Step Latency
    bars2 = ax2.bar(frameworks, step_times, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8)
    ax2.set_title("Optimization Step Latency", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Microseconds (µs) / Step", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max(step_times) * 0.02), f"{yval:.1f} µs", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, max(step_times) * 1.15)

    plt.suptitle("CPU Computational Performance Benchmark", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p2 = os.path.join(assets_dir, "benchmark_throughput.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # 3. Peak RAM Memory Usage (RSS)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    mem_vals = [numpy_res["peak_memory_mb"], torch_res["peak_memory_mb"]]
    bars3 = ax.bar(frameworks, mem_vals, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8)
    ax.set_title("Peak Process Memory Usage (RSS)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Memory (MB)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + (max(mem_vals) * 0.02), f"{yval:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(mem_vals) * 1.18)

    plt.tight_layout()
    p3 = os.path.join(assets_dir, "benchmark_memory.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    generate_benchmark_plots()
