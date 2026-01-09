# PhyPlan Monorepo

This repository contains two independent parts to avoid mixing files in the same folder:

- `ECCV/`: (renamed from the original local `ICML/`) script-based planning/QA data generation and post-processing.
- `Qwen-PC/`: Qwen3-VL + PointLLM related code, demos, finetune/eval tooling and checkpoints.

## Quick Start

### ECCV

- Run a generator: `python ECCV/mani_longvideo.py`
- Run API-based generator: `API_KEY=... API_BASE_URL=... python ECCV/generate_phyplan_api.py`

### Qwen-PC

See `Qwen-PC/README.md`.

