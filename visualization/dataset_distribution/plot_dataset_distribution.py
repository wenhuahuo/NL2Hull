#!/usr/bin/env python3
"""Create the dataset-distribution panels for the manuscript.

Panel (a) reuses the archived Qwen3-Embedding PCA coordinates and joins each
point to its structured action record. Panel (b) counts the six typed-question
families in the complete SDD Dataset and SDDBench. No data are refit or
resampled except for the fixed six-component MiniBatchKMeans annotation layer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

MODELS = (
    "Claude Opus 5", "Kimi K3", "GLM 5.3", "Grok 4.7",
    "DeepSeek Flash", "GPT 5.6", "MiMo V2.6 Pro",
)
# Palette retained from the original source-model plot.
MODEL_COLORS = (
    "#426897", "#F0A450", "#62A895", "#BD789D",
    "#C75D59", "#8C77AF", "#92A753",
)
MODEL_KEYS = (
    "wokey/claude-opus-5", "wokey/kimi-k3", "wokey/glm-5.3",
    "wokey/grok-4.7", "deepseek/deepseek-flash", "wokey/gpt-5.6-sol",
    "openrouter/xiaomi/mimo-v2.6-pro",
)
QUESTION_KEYS = (
    "ship_action_count", "ship_region", "ship_operation",
    "ship_magnitude_mode", "ship_magnitude_level", "ship_constraint",
)
QUESTION_LABELS = (
    "Action count", "Region", "Operation", "Magnitude\nmode",
    "Magnitude\nlevel", "Constraint",
)
CLUSTER_LABELS = {
    "long_context": "Long, context-rich\nengineering prompts",
    "standard_context": "Standard hull-context\nrequests",
    "explicit_numeric": "Explicit numeric\nrange/magnitude",
    "short_multi": "Short multi-action\nrequests",
    "short_direct": "Short direct or\nfollow-up requests",
    "midbody_bilge": "Midbody/bilge-focused\nfollow-up requests",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--projection", type=Path, required=True)
    p.add_argument("--language", type=Path, required=True)
    p.add_argument("--structured", type=Path, required=True)
    p.add_argument("--sdd-files", type=Path, nargs="+", required=True)
    p.add_argument("--bench-file", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "figure8")
    return p.parse_args()


def load_records(language: Path, structured: Path, xy: np.ndarray) -> list[dict]:
    source = {}
    with structured.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            source[record["sample_id"]] = record
    records = []
    with language.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # The archived projection intentionally excludes synthetic rollback
            # text, matching prepare_and_embed.py.
            if item["model"] == "synthetic_rollback_v1":
                continue
            record = source[item["sample_id"]]
            actions = [record["actions"][i] for i in item["action_indices"]]
            records.append({
                "item": item,
                "actions": actions,
                "text_length": len(item["text"]),
            })
    if len(records) != len(xy):
        raise ValueError(f"projection rows={len(xy)} but joined records={len(records)}")
    return records


def cluster_annotations(xy: np.ndarray, records: list[dict]) -> tuple[np.ndarray, dict]:
    labels = MiniBatchKMeans(
        n_clusters=6, n_init=20, batch_size=4096,
        max_iter=200, random_state=20260925,
    ).fit_predict(xy)
    centers = np.array([xy[labels == c].mean(axis=0) for c in range(6)])
    order = sorted(range(6), key=lambda c: centers[c, 0])
    # The left-to-right order is stable for this archived projection. Names are
    # based on representative texts, text length, interaction composition, and
    # action-region composition; they are descriptive annotations, not labels
    # used during embedding or clustering.
    names = (
        "long_context", "standard_context", "explicit_numeric",
        "short_multi", "short_direct", "midbody_bilge",
    )
    cluster_info = {}
    interactions = [r["item"]["interaction_type"] for r in records]
    models = [r["item"]["model"] for r in records]
    for position, cluster in enumerate(order):
        indices = np.flatnonzero(labels == cluster)
        cluster_info[names[position]] = {
            "cluster_id": int(cluster),
            "center": centers[cluster].round(6).tolist(),
            "count": int(len(indices)),
            "mean_text_characters": float(np.mean([records[i]["text_length"] for i in indices])),
            "interaction_counts": dict(Counter(interactions[i] for i in indices)),
            "model_counts": dict(Counter(models[i] for i in indices)),
        }
    ami_interaction = adjusted_mutual_info_score(interactions, labels)
    ami_model = adjusted_mutual_info_score(models, labels)
    return labels, {
        "algorithm": "MiniBatchKMeans",
        "clusters": 6,
        "random_state": 20260925,
        "adjusted_mutual_information_interaction_type": float(ami_interaction),
        "adjusted_mutual_information_source_model": float(ami_model),
        "cluster_info": cluster_info,
    }


def plot_pca(xy: np.ndarray, model_ids: np.ndarray, output: Path,
             records: list[dict], analysis: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": .65,
    })
    fig = plt.figure(figsize=(7.2, 4.9), facecolor="white")
    ax = fig.add_axes((.09, .11, .68, .80))
    rng = np.random.default_rng(20260925)
    # Interleave source models before rasterization so the last model in the
    # legend cannot visually cover the other six colors.
    order = rng.permutation(len(xy))
    point_colors = np.asarray(MODEL_COLORS, dtype=object)[model_ids]
    ax.scatter(xy[order, 0], xy[order, 1], s=6.2, alpha=.34,
               color=point_colors[order], edgecolors="none", rasterized=True)
    ax.set_xlabel("PCA 1", labelpad=2)
    ax.set_ylabel("PCA 2", labelpad=2)
    ax.tick_params(length=2.5, pad=2)
    ax.set_xlim(-.62, .32)
    ax.set_ylim(-.72, .34)
    ax.grid(color="#D9DDE2", lw=.35, alpha=.55)
    ax.set_axisbelow(True)

    # Arrow targets are the archived six-cluster centroids, ordered left to right.
    target = {name: np.asarray(info["center"])
              for name, info in analysis["cluster_info"].items()}
    placements = {
        "long_context": ((-.59, .30), "left", "bottom"),
        "standard_context": ((-.34, .31), "left", "bottom"),
        "explicit_numeric": ((-.31, -.68), "left", "top"),
        "short_multi": ((-.01, .31), "left", "bottom"),
        "short_direct": ((.30, .17), "right", "bottom"),
        "midbody_bilge": ((.31, -.67), "right", "top"),
    }
    for name, (text_xy, ha, va) in placements.items():
        ax.annotate(
            CLUSTER_LABELS[name], xy=target[name], xycoords="data",
            xytext=text_xy, textcoords="data", ha=ha, va=va,
            fontsize=6.0, linespacing=1.02, color="#263238",
            bbox={"boxstyle": "round,pad=.26", "fc": "white", "ec": "#8A949E", "lw": .55, "alpha": .94},
            arrowprops={"arrowstyle": "-|>", "color": "#455A64", "lw": .7,
                        "shrinkA": 3, "shrinkB": 2, "connectionstyle": "arc3,rad=.12"},
        )
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                       markeredgecolor="none", markersize=4.5, label=name)
               for name, color in zip(MODELS, MODEL_COLORS)]
    ax.legend(handles=handles, title="Generator", loc="lower left",
              bbox_to_anchor=(.018, .018), ncol=2, frameon=False, fontsize=5.4,
              title_fontsize=5.8, borderaxespad=0., handletextpad=.35,
              columnspacing=.8, labelspacing=.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def question_counts(paths: list[Path]) -> tuple[Counter, int]:
    counts = Counter()
    records = 0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                records += 1
                for question in item["questions"].values():
                    counts[question["src"]] += 1
    return counts, records


def plot_radar(sdd: Counter, sdd_records: int, bench: Counter,
               bench_records: int, output: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7})
    values_sdd = np.array([sdd[key] for key in QUESTION_KEYS], dtype=float)
    values_bench = np.array([bench[key] for key in QUESTION_KEYS], dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(QUESTION_KEYS), endpoint=False)
    angles = np.r_[angles, angles[0]]
    values_sdd = np.r_[values_sdd, values_sdd[0]]
    values_bench = np.r_[values_bench, values_bench[0]]
    # The taller canvas compensates for the narrower right-hand subfigure so
    # the visible radar panel has the same rendered height as the PCA panel.
    fig = plt.figure(figsize=(4.8, 6.0), facecolor="white")
    ax = fig.add_axes((.06, .08, .88, .81), polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values_sdd, color="#D97706", lw=1.6)
    ax.fill(angles, values_sdd, color="#F59E0B", alpha=.18)
    ax.plot(angles, values_bench, color="#1565C0", lw=1.6)
    ax.fill(angles, values_bench, color="#3B82F6", alpha=.28)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(QUESTION_LABELS, fontsize=7)
    ax.tick_params(axis="x", pad=8)
    # Absolute question counts are retained, while a logarithmic radial scale
    # keeps the much smaller SDDBench polygon visible without rescaling it.
    ax.set_yscale("log")
    ax.set_ylim(1e3, max(values_sdd) * 1.12)
    radial_ticks = np.array([1e3, 1e4, 1e5, 3e5])
    radial_ticks = radial_ticks[radial_ticks < max(values_sdd) * 1.12]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(["1k", "10k", "100k", "300k"][:len(radial_ticks)], fontsize=5.8)
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", pad=2, length=2)
    ax.grid(color="#C9D1D9", lw=.45, alpha=.8)
    ax.spines["polar"].set_color("#8A949E")
    ax.spines["polar"].set_linewidth(.65)
    ax.legend([
        Line2D([0], [0], color="#D97706", lw=1.7),
        Line2D([0], [0], color="#1565C0", lw=1.7),
    ], [f"SDD Dataset (n={sum(sdd.values())/1e6:.3f}M)",
        f"SDDBench (n={sum(bench.values()):,})"],
        loc="lower left", bbox_to_anchor=(.025, .025), frameon=True,
        facecolor="white", framealpha=.82, edgecolor="none",
        fontsize=6.0, handlelength=1.6, borderaxespad=.2,
        labelspacing=.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    projection = np.load(args.projection)
    xy, model_ids = projection["xy"], projection["model"]
    records = load_records(args.language, args.structured, xy)
    _, analysis = cluster_annotations(xy, records)
    plot_pca(xy, model_ids, args.output / "pca.png", records, analysis)
    sdd_counts, sdd_records = question_counts(args.sdd_files)
    bench_counts, bench_records = question_counts([args.bench_file])
    plot_radar(sdd_counts, sdd_records, bench_counts, bench_records, args.output / "radar.png")
    manifest = {
        "projection_sha256": hashlib.sha256(args.projection.read_bytes()).hexdigest(),
        "projection_rows": int(len(xy)),
        "sdd_records": sdd_records,
        "sdd_question_counts": dict(sdd_counts),
        "sddbench_records": bench_records,
        "sddbench_question_counts": dict(bench_counts),
        "question_order": list(QUESTION_KEYS),
        "cluster_analysis": analysis,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
