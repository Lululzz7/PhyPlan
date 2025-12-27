# Repository Guidelines

## Project Structure & Module Organization

- `*.py` (repo root): standalone Python scripts for generating and post-processing planning/QA data (e.g., `mani_longvideo.py`, `nav_high_videos.py`, `generate_phyplan_api*.py`, `extract_*_segments.py`).
- `causal_spafa_plan_dataset*/*`: generated dataset outputs (JSONL + per-task folders, sampled frames, segment exports). Treat as artifacts, not source.
- `generated_plans_output_*/*`: additional generated plan outputs and sampled frames.
- `*_spec.md`, `mani_*_tasks_plan*.md`: task specs/notes used to guide generation.

## Build, Test, and Development Commands

This repository is script-driven (no `Makefile`/`pyproject.toml`). Use a Python venv.

- Create env (example): `python -m venv .venv && source .venv/bin/activate`
- Common deps (based on imports): `pip install openai opencv-python numpy`
- Run a generator (example): `python mani_longvideo.py` (edit `ScriptConfig` paths/limits inside the file)
- Run API-based generators (env-driven example): `API_KEY=... API_BASE_URL=... python generate_phyplan_api.py`
- Quick sanity check: `python -m compileall .`

## Coding Style & Naming Conventions

- Python: 4-space indentation, UTF-8 files, prefer type hints + `@dataclass` for schemas.
- Keep scripts runnable: avoid introducing hard package-level imports when optional deps are used (see `cv2` guarded import patterns).
- Naming: follow existing patterns (`mani_*video*.py` for manipulation, `nav_*` for navigation, `generate_*` for dataset generation).

## Testing Guidelines

- No formal unit test suite is included.
- Validate changes by running the smallest relevant script on a small input and confirming expected artifacts (e.g., `data.jsonl`, `sampled_frames/`).

## Commit & Pull Request Guidelines

- This checkout does not include Git history; use clear, imperative commit subjects (e.g., “Fix API response parsing”).
- Do not commit large generated artifacts (e.g., `causal_spafa_plan_dataset*`, `generated_plans_output_*`) unless the PR explicitly updates a published dataset.
- PRs should include: purpose, the script(s) affected, how to reproduce (exact command + key config values), and a small sample output path.

## Security & Configuration Tips

- Never hardcode or commit API keys. Prefer environment variables (`API_KEY`, `API_BASE_URL`, `MODEL_PROVIDER_ID`, `MODEL_NAME`) where supported.
- Generated outputs can be large; write to dedicated output folders and keep runs clearly labeled.
