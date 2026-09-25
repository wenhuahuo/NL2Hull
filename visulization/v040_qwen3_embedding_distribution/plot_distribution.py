"""Plot an equal-size per-model sample of a fitted 2-D embedding projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DISPLAY = (
    'Claude Opus 5', 'Kimi K3', 'GLM 5.3', 'Grok 4.7',
    'DeepSeek Flash', 'GPT 5.6', 'MiMo V2.6 Pro',
)
COLORS = ('#426897', '#F0A450', '#62A895', '#BD789D',
          '#C75D59', '#8C77AF', '#92A753')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points-per-model', type=int, default=3000)
    args = parser.parse_args()
    data = np.load(args.projection)
    xy, models = data['xy'], data['model']
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest['model_order']) == len(DISPLAY) == len(COLORS)
    assert len(xy) == len(models) == manifest['sentence_count']
    rng = np.random.default_rng(manifest['random_seed'])
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9.0, 6.4), facecolor='white')
    for index, (name, color) in enumerate(zip(DISPLAY, COLORS)):
        subset = np.flatnonzero(models == index)
        chosen = rng.choice(subset, min(args.points_per_model, len(subset)), replace=False)
        ax.scatter(xy[chosen, 0], xy[chosen, 1], s=3.3, alpha=.28,
                   color=color, edgecolors='none', rasterized=True, label=name)
    ax.set(title='Generated FFD descriptions by source model', xlabel='PCA 1', ylabel='PCA 2')
    ax.legend(title='Generator', loc='upper left', bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=8, title_fontsize=9, markerscale=3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
