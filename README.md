# NL2Hull

Natural-language-driven constrained ship-form design with NURBS and free-form deformation (FFD).

[中文说明](README.zh-CN.md)

[![arXiv](https://img.shields.io/badge/arXiv-2610.09896-b31b1b.svg)](https://arxiv.org/abs/2610.09896)
[![Chip-0.8B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Chip--0.8B-yellow)](https://huggingface.co/wenhuahuo/chip-0.8b)
[![Ship Design Decision Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Ship%20Design%20Decisions-blue)](https://huggingface.co/datasets/wenhuahuo/ship-design-decisions)

![](assets/nl2hull_framework.png)

## Overview

NL2Hull discretizes ship-design language and introduces a [Jev-like](https://github.com/jaredpalmer/kev) decision model to map semantics to executable geometric operations. The pipeline combines:

- normalized NURBS representations for ship waterlines and longitudinal profiles;
- typed decisions over hull regions, operations, magnitude levels, and preservation constraints;
- FFD action projection and sequential control-point deformation;
- hull reconstruction, geometric validity checks, and constraint evaluation;
- shared benchmarks for decision accuracy, probability calibration, action exact match, and end-to-end execution.

The repository provides the geometric engine, structured-action data pipeline, decision-interface adapters, evaluation scripts, tests, and visualization tools used by the project.

## Method

A design request is represented as an ordered sequence of typed decisions. A Jev-style decision interface selects the action count and action attributes, then the FFD engine applies the selected operations to a normalized NURBS hull.

```text
Natural-language request
        ↓
typed region / operation / magnitude / constraint decisions
        ↓
ordered FFD actions
        ↓
NURBS control-point deformation
        ↓
hull reconstruction and constraint checks
```

The geometric representation uses cubic NURBS waterlines, fore and aft longitudinal profiles, symmetry-preserving section construction, and smooth local influence windows. The evaluation protocol records failed calls, invalid probabilities, malformed actions, and geometry failures in the relevant denominators.

![](assets/end2end_single.png)

## Dataset

The project uses twelve normalized classic hull geometries and builds the SDD Dataset (Ship Design Decision Dataset) and SDDBench, with the following data directories:

- `datasets/classic_hulls/`: source and canonical hull geometries with metadata and benchmark manifests;
- `datasets/ship_design_structured_actions/`: 30,000 programmatically generated structured FFD actions;
- `datasets/ship_design_decisions/`: 134,558 cleaned language–decision records with train, validation, test, and SDDBench splits;
- `datasets/jevbench/`: the pinned [JevBench](https://github.com/fstandhartinger/jevbench) used by the comparison protocol;
- `datasets/kev_decision_v7/`: the pinned upstream Kev decision-v7 suite used for the original pretraining stage.

The ship-design decision corpus contains 88,604 training records, 22,758 validation records, and 23,196 test records. Its splits are separated by hull family: training uses DTC, DTMB 5415, KCS, KVLCC2, containership, frigate, and NPL variants; validation uses S-175 and Series 60; testing uses Wigley and workboat hulls.

![](assets/dataset_distribution_radar.png)

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run the classic-hull reconstruction benchmark:

```bash
python -m nurbs_ship_reconstruction.cli \
  --dataset datasets/classic_hulls \
  --output outputs/local_reconstruction
```

Run a focused smoke test:

```bash
python -m nurbs_ship_reconstruction.cli \
  --dataset datasets/classic_hulls \
  --hull wigley_hull \
  --levels 12 \
  --output outputs/wigley_smoke
```

Run the automated tests:

```bash
python -m pytest
```

## Repository layout

```text
src/nurbs_ship_reconstruction/  NURBS representation, reconstruction, and FFD engine
src/dataset/                    structured and language dataset preparation
src/benchmarks/                 common action and probability evaluation
src/end2end/                    action execution and end-to-end evaluation
src/model_clients/              decision-service and local-model adapters
scripts/evaluate/               evaluation entry points
scripts/slurm/                  cluster training and evaluation jobs
tests/                          unit and integration tests
visualization/                  paper and diagnostic figure generation
datasets/                       local data and manifests
outputs/                        local experiment outputs
```

## Citation

- Paper: [NL2Hull on arXiv](https://arxiv.org/abs/2610.09896)
- Model: [wenhuahuo/chip-0.8b](https://huggingface.co/wenhuahuo/chip-0.8b)
- Dataset: [wenhuahuo/ship-design-decisions](https://huggingface.co/datasets/wenhuahuo/ship-design-decisions)

```bibtex
@article{huo2026nl2hull,
  title         = {NL2Hull: A Natural Language-Driven Constrained Ship Design Decision Framework},
  author        = {Huo, Wenhua and Han, Fenglei and Zhao, Wangyuan and Wu, Jialin and Han, Jiayi},
  year          = {2026},
  eprint        = {2610.09896},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.09896}
}
```
