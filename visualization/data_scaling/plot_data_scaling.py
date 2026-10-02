#!/usr/bin/env python3
"""Plot the training-data scaling result used in Section 3.4.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = (
    (0, "step0_eval5k/report.json"),
    (1, "frac1/eval5k/report.json"),
    (2, "frac2/eval5k/report.json"),
    (4, "frac4/eval5k/report.json"),
    (8, "frac8/eval5k/report.json"),
    (25, "frac25/eval5k/report.json"),
)


def read_metrics(data_dir: Path, full_report: Path) -> tuple[list[float], list[float], list[float]]:
    shares, question_accuracy, ffd_exact_match = [], [], []
    for share, relative_path in RUNS:
        report = json.loads((data_dir / relative_path).read_text())
        shares.append(share)
        question_accuracy.append(100 * report["clean"]["acc"])
        ffd_exact_match.append(100 * report["ffd_accuracy"]["turn_exact_rate"])
    report = json.loads(full_report.read_text())
    shares.append(100)
    question_accuracy.append(100 * report["clean"]["acc"])
    ffd_exact_match.append(100 * report["ffd_accuracy"]["turn_exact_rate"])
    return shares, question_accuracy, ffd_exact_match


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=root / "outputs/data_scaling_kev08")
    parser.add_argument("--full-report", type=Path,
                        default=root / "outputs/v045_kev08_test5k_eval/report.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "figure_data_scaling.png")
    args = parser.parse_args()
    shares, question_accuracy, ffd_exact_match = read_metrics(args.data_dir, args.full_report)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": .65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    fig, ax = plt.subplots(figsize=(6.4, 3.55), facecolor="white")
    ax.plot(shares, question_accuracy, color="#426897", marker="o", ms=4.0,
            lw=1.35, label="Question accuracy")
    ax.plot(shares, ffd_exact_match, color="#D97706", marker="s", ms=3.8,
            lw=1.35, label="FFD exact match")
    ax.set_xlabel("Training data share (%)", labelpad=2)
    ax.set_ylabel("Accuracy (%)", labelpad=2)
    ax.set_xlim(-2, 103)
    ax.set_ylim(70, 100.5)
    ax.set_xticks(shares)
    ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
    ax.grid(axis="y", color="#D9DDE2", lw=.4, alpha=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.6,
              handlelength=1.7, labelspacing=.35)
    fig.tight_layout(pad=.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
