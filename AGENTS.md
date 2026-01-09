# Repository Guidelines

## Project Structure & Module Organization
- Core: `Qwen3-VL/` (vision–language code), `PointLLM/pointllm/`, `qwen-vl-utils/src/qwen_vl_utils/`.
- Demos: `web_demo_mm.py`, `qwen-vl-finetune/demo/`.
- Evaluation: `evaluation/mmmu/` (scripts, utils, prompts).
- Docs & tooling: `docs/`, `docker/`, model `checkpoints/`.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install packages:
  - Utils: `pip install -e qwen-vl-utils`
  - PointLLM: `pip install -e PointLLM`
  - Web demo deps: `pip install -r requirements_web_demo.txt`
- Run demo: `python web_demo_mm.py`.
- Evaluate MMMU: `python evaluation/mmmu/run_mmmu.py`.

## Coding Style & Naming Conventions
- Python, 4‑space indentation; PEP8 defaults.
- Names: `snake_case` for functions/variables, `PascalCase` for classes; files like `run_mmmu.py`.
- Lint/format with Ruff (see `qwen-vl-utils/pyproject.toml`):
  - `ruff check qwen-vl-utils/src`
  - `ruff format qwen-vl-utils/src`
- Keep line length ≤119; organize imports per configured isort rules.

## Testing Guidelines
- Prefer `pytest` tests under `tests/` near modules; name files `test_*.py`.
- For evaluation-style checks, add scripts under `evaluation/` and document dataset paths.
- Run tests: `pytest -q` (if present) or module-specific scripts (e.g., MMMU above).

## Commit & Pull Request Guidelines
- Commit subject in imperative mood; include scope: `feat(utils): add smart_resize`.
- Use conventional types: `feat`, `fix`, `docs`, `refactor`, `test`.
- PRs include description, linked issues, reproduction commands, and sample outputs/screenshots if UI/demo changes.

## Security & Configuration Tips
- Never commit secrets; set `DASHSCOPE_API_KEY` in the environment.
- Store large artifacts in `checkpoints/`; prefer Git LFS or external storage.
