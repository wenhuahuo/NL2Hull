---
title: NL2Hull Demo
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

# NL2Hull Demo

This Space demonstrates the end-to-end NL2Hull pipeline on a KVLCC2 hull. It loads the public Chip-0.8B model, predicts typed ship-design decisions from a Chinese natural-language request, applies the resulting FFD actions to the NURBS hull representation, and reports geometric validity and constraint metrics. The interface presents original and modified models in a two-column, three-row layout with draggable 3D views, side views, top views, and an optional original-model diff overlay.

- Paper: [arXiv:2610.09896](https://arxiv.org/abs/2610.09896)
- Code: [wenhuahuo/NL2Hull](https://github.com/wenhuahuo/NL2Hull)
- Model: [wenhuahuo/chip-0.8b](https://huggingface.co/wenhuahuo/chip-0.8b)
- Dataset: [wenhuahuo/ship-design-decisions](https://huggingface.co/datasets/wenhuahuo/ship-design-decisions)
