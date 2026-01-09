# Repository Guidelines

## Project Structure & Module Organization
### ECCV (original ICML scripts)
- Location: `ECCV/`
- `ECCV/*.py`: standalone Python scripts for generating and post-processing planning/QA data (e.g., `mani_longvideo.py`, `nav_high_videos.py`, `generate_phyplan_api*.py`, `extract_*_segments.py`).
- `causal_spafa_plan_dataset*/*`: generated dataset outputs (JSONL + per-task folders, sampled frames, segment exports). Treat as artifacts, not source.
- `generated_plans_output_*/*`: additional generated plan outputs and sampled frames.
- `ECCV/*_spec.md`, `ECCV/mani_*_tasks_plan*.md`: task specs/notes used to guide generation.

### Qwen-PC (Qwen3-VL + PointLLM)
- Location: `Qwen-PC/`
- Core: `Qwen-PC/qwen-vl-finetune/`, `Qwen-PC/PointLLM/pointllm/`, `Qwen-PC/qwen-vl-utils/src/qwen_vl_utils/`.
- Demos: `Qwen-PC/web_demo_mm.py`, `Qwen-PC/qwen-vl-finetune/demo/`.
- Evaluation: `Qwen-PC/evaluation/mmmu/` (scripts, utils, prompts).
- Docs & tooling: `Qwen-PC/docs/`, `Qwen-PC/docker/`, model `Qwen-PC/checkpoints/`.

## Build, Test, and Development Commands

### ECCV
- Create env (example): `python -m venv .venv && source .venv/bin/activate`
- Common deps (based on imports): `pip install openai opencv-python numpy`
- Run a generator (example): `python ECCV/mani_longvideo.py` (edit `ScriptConfig` paths/limits inside the file)
- Run API-based generators (example): `API_KEY=... API_BASE_URL=... python ECCV/generate_phyplan_api.py`
- Quick sanity check: `python -m compileall ECCV`

### Qwen-PC
- Create venv: `python -m venv .venv && source .venv/bin/activate` (recommended to run inside `Qwen-PC/`).
- Install packages:
  - Utils: `pip install -e Qwen-PC/qwen-vl-utils`
  - PointLLM: `pip install -e Qwen-PC/PointLLM`
  - Web demo deps: `pip install -r requirements_web_demo.txt`
- Run demo: `python Qwen-PC/web_demo_mm.py`.
- Evaluate MMMU: `python Qwen-PC/evaluation/mmmu/run_mmmu.py`.

## Coding Style & Naming Conventions

- Python: 4-space indentation; UTF-8 files.
- ECCV scripts: prefer type hints + `@dataclass` for schemas; keep scripts runnable (optional deps guarded imports like `cv2`).
- Qwen-PC: PEP8 defaults; keep line length ≤119; organize imports per configured isort rules.
- Lint/format with Ruff (see `Qwen-PC/qwen-vl-utils/pyproject.toml`):
  - `ruff check Qwen-PC/qwen-vl-utils/src`
  - `ruff format Qwen-PC/qwen-vl-utils/src`
- Naming: follow existing patterns (`mani_*video*.py` for manipulation, `nav_*` for navigation, `generate_*` for dataset generation); `snake_case` for functions/variables, `PascalCase` for classes.

## Testing Guidelines

- ECCV: no formal unit test suite; validate by running the smallest relevant script on a small input and confirming expected artifacts (e.g., `data.jsonl`, `sampled_frames/`).
- Qwen-PC: prefer `pytest` tests under `tests/` near modules; name files `test_*.py`; run `pytest -q` (if present) or module-specific scripts (e.g., MMMU above).

## Commit & Pull Request Guidelines

- Use clear, imperative commit subjects; conventional types/scopes are welcome (e.g., `feat(utils): add smart_resize`).
- Do not commit large generated artifacts (e.g., `causal_spafa_plan_dataset*`, `generated_plans_output_*`) unless explicitly updating a published dataset.
- PRs should include: purpose, the script(s)/modules affected, how to reproduce (exact command + key config values), and a small sample output path.

## Security & Configuration Tips

- Never hardcode or commit API keys/secrets. Prefer environment variables (e.g., `API_KEY`, `API_BASE_URL`, `MODEL_PROVIDER_ID`, `MODEL_NAME`, `DASHSCOPE_API_KEY`) where supported.
- Generated outputs can be large; write to dedicated output folders and keep runs clearly labeled.
- Store large model artifacts in `checkpoints/`; prefer Git LFS or external storage when appropriate.
