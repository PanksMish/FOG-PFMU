"""
Paper Figure Reproduction
=========================
Generates Figures 2–23 from the paper using collected simulation results.

Usage:
    from src.visualization.plots import PaperPlots
    pp = PaperPlots(results_dir="results")
    pp.plot_all(data)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# ---- Styling ----
METHODS = ["CLD", "F-RL", "A-MORL", "HCF", "F-PARK", "F-IPS", "PFMU"]
COLORS = {
    "CLD":    "#d62728",
    "F-RL":   "#1f77b4",
    "A-MORL": "#2ca02c",
    "HCF":    "#ff7f0e",
    "F-PARK": "#9467bd",
    "F-IPS":  "#8c564b",
    "PFMU":   "#e377c2",
}
MARKERS = {
    "CLD": "s", "F-RL": "^", "A-MORL": "D",
    "HCF": "o", "F-PARK": "v", "F-IPS": "P", "PFMU": "*"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


class PaperPlots:
    """Generates all paper figures from experiment result dictionaries."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.fig_dir = os.path.join(results_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        path = os.path.join(self.fig_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Figure 2 — Signal Response Latency (mean ± std)
    # ------------------------------------------------------------------
    def fig02_latency(self, data: Dict):
        """Bar chart of signal response latency per method (Fig. 2)."""
        methods = [m for m in METHODS if m in data]
        means = [data[m]["latency_s"]["mean"] for m in methods]
        stds  = [data[m]["latency_s"]["std"] for m in methods]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(methods, means, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.7)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Signal Response Latency (s)")
        ax.set_title("Fig. 2 — Signal Response Latency (mean ± std)")
        ax.set_ylim(0, max(means) * 1.3)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig02_latency.png")

    # ------------------------------------------------------------------
    # Figure 3 — Intersection Throughput
    # ------------------------------------------------------------------
    def fig03_throughput(self, data: Dict):
        methods = [m for m in METHODS if m in data]
        means = [data[m]["throughput_vph"]["mean"] for m in methods]
        stds  = [data[m]["throughput_vph"]["std"] for m in methods]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(methods, means, xerr=stds, capsize=4,
                color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.7)
        for i, (mean, method) in enumerate(zip(means, methods)):
            ax.text(mean + 5, i, f"{mean:.0f}", va="center", fontsize=9)
        ax.set_xlabel("Intersection Throughput (veh/h)")
        ax.set_title("Fig. 3 — Intersection Throughput (mean ± std)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig03_throughput.png")

    # ------------------------------------------------------------------
    # Figure 4 — Jain's Fairness Index
    # ------------------------------------------------------------------
    def fig04_fairness(self, data: Dict):
        methods = [m for m in METHODS if m in data]
        means = [data[m]["fairness_index"]["mean"] for m in methods]
        stds  = [data[m]["fairness_index"]["std"] for m in methods]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(methods, means, xerr=stds, capsize=4,
                color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.7)
        ax.set_xlabel("Jain's Fairness Index")
        ax.set_title("Fig. 4 — Fairness Index across Intersections")
        ax.set_xlim(0.75, 1.0)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig04_fairness.png")

    # ------------------------------------------------------------------
    # Figure 6 — Learning Convergence
    # ------------------------------------------------------------------
    def fig06_convergence(self, training_data: Dict):
        """Line plot of average throughput over training episodes."""
        fig, ax = plt.subplots(figsize=(8, 5))
        methods_to_plot = ["F-RL", "A-MORL", "HCF", "PFMU"]
        for method in methods_to_plot:
            if method not in training_data:
                continue
            episodes = training_data[method]["episodes"]
            mean_tp  = training_data[method]["mean_throughput"]
            std_tp   = training_data[method].get("std_throughput", np.zeros_like(mean_tp))
            ax.plot(episodes, mean_tp, label=method, color=COLORS[method],
                    marker=MARKERS[method], markevery=50, linewidth=2)
            ax.fill_between(episodes, mean_tp - std_tp, mean_tp + std_tp,
                            alpha=0.15, color=COLORS[method])
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Average Intersection Throughput (veh/h)")
        ax.set_title("Fig. 6 — Learning Convergence with Confidence Bands")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig06_convergence.png")

    # ------------------------------------------------------------------
    # Figure 9 — Latency Scaling
    # ------------------------------------------------------------------
    def fig09_scalability(self, scale_data: Dict):
        """Latency vs number of intersection nodes (CLD vs HCF vs PFMU)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        methods_to_plot = ["CLD", "HCF", "PFMU"]
        for method in methods_to_plot:
            if method not in scale_data:
                continue
            sizes = scale_data[method]["sizes"]
            latencies = scale_data[method]["latencies"]
            stds = scale_data[method].get("stds", np.zeros_like(latencies))
            ax.plot(sizes, latencies, label=method, color=COLORS[method],
                    marker=MARKERS[method], linewidth=2)
            ax.fill_between(sizes, np.array(latencies) - np.array(stds),
                            np.array(latencies) + np.array(stds),
                            alpha=0.2, color=COLORS[method])
        ax.set_xlabel("Number of Intersection Nodes")
        ax.set_ylabel("Average Signal Response Latency (s)")
        ax.set_title("Fig. 9 — Latency Scaling with Increasing Network Size")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig09_scalability.png")

    # ------------------------------------------------------------------
    # Figure 13 — Runtime Comparison
    # ------------------------------------------------------------------
    def fig13_runtime(self, data: Dict):
        methods = [m for m in METHODS if m in data]
        means = [data[m]["runtime_ms"]["mean"] for m in methods]
        stds  = [data[m]["runtime_ms"]["std"] for m in methods]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(methods, means, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.7)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{mean:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Computation Runtime (ms)")
        ax.set_title("Fig. 13 — Runtime Comparison across Benchmark Methods")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig13_runtime.png")

    # ------------------------------------------------------------------
    # Figure 14 — Latency CDF
    # ------------------------------------------------------------------
    def fig14_latency_cdf(self, latency_samples: Dict):
        """Empirical CDF of signal response latency (Fig. 14)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, samples in latency_samples.items():
            sorted_s = np.sort(samples)
            cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
            ax.plot(sorted_s, cdf, label=method, color=COLORS.get(method, "grey"),
                    linestyle="-" if method == "PFMU" else "--", linewidth=2)
        ax.set_xlabel("Signal Response Latency (s)")
        ax.set_ylabel("Cumulative Distribution Function")
        ax.set_title("Fig. 14 — CDF of Signal Response Latency")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fig14_latency_cdf.png")

    # ------------------------------------------------------------------
    # Figure 15 — Ablation Study
    # ------------------------------------------------------------------
    def fig15_ablation(self, ablation_data: Dict):
        """Bar chart comparing Full PFMU vs component ablations (Fig. 15)."""
        variants = ["Full PFMU", "w/o Coord.", "w/o Parking", "w/o Fog"]
        latencies = [ablation_data[v]["latency_s"] for v in variants]
        throughputs = [ablation_data[v]["throughput_vph"] for v in variants]

        x = np.arange(len(variants))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        bars1 = ax1.bar(x - width/2, latencies, width, label="Latency (s)",
                        color="#1f77b4", alpha=0.8, hatch="//")
        bars2 = ax2.bar(x + width/2, throughputs, width, label="Throughput (veh/h)",
                        color="#ff7f0e", alpha=0.8, hatch="\\\\")

        ax1.set_ylabel("Signal Response Latency (s)")
        ax2.set_ylabel("Intersection Throughput (veh/h)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants)
        ax1.set_title("Fig. 15 — Ablation Study: Impact of Key Components")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        self._save(fig, "fig15_ablation.png")

    # ------------------------------------------------------------------
    # Plot All
    # ------------------------------------------------------------------
    def plot_all(self, results: Dict, training_data: Dict = None,
                 scale_data: Dict = None, latency_samples: Dict = None,
                 ablation_data: Dict = None):
        """Generate all available plots."""
        self.fig02_latency(results)
        self.fig03_throughput(results)
        self.fig04_fairness(results)
        self.fig13_runtime(results)
        if training_data:
            self.fig06_convergence(training_data)
        if scale_data:
            self.fig09_scalability(scale_data)
        if latency_samples:
            self.fig14_latency_cdf(latency_samples)
        if ablation_data:
            self.fig15_ablation(ablation_data)
        logger.info(f"All figures saved to {self.fig_dir}")
