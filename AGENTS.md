# Repository Guidelines

## Project Structure & Module Organization
`src/lerobot/` contains the main package (robots, teleoperators, envs, datasets, RL, optimization, utilities, and CLI scripts). `tests/` mirrors product areas (`tests/datasets`, `tests/robots`, `tests/rl`, `tests/training`, etc.). Use `examples/` for runnable reference workflows, `docs/source/` for user documentation, `docker/` for container builds, `benchmarks/` for perf scripts, and `media/` for documentation assets.

## Build, Test, and Development Commands
Use Python 3.10.

```bash
uv sync --extra "dev,test"        # install project with dev/test extras
pre-commit install                # enable local hooks
pre-commit run --all-files        # run lint/format/security/type checks
uv run pytest tests -vv --maxfail=10
uv run make test-end-to-end       # end-to-end CLI train/eval smoke tests
make build-user                   # build docker/Dockerfile.user
make build-internal               # build docker/Dockerfile.internal
```

If you do not use `uv`, use `pip install -e ".[dev,test]"`.

## Coding Style & Naming Conventions
Formatting and lint rules come from `pyproject.toml` and pre-commit. Key defaults: Ruff, Python 3.10, max line length 110, and double-quoted strings. Prefer snake_case for modules/functions/variables, PascalCase for classes, and explicit, scope-aware names (for example `config_so_follower.py`, `test_actor_learner.py`). Keep CLI-facing behavior in `src/lerobot/scripts/`.

## Testing Guidelines
Testing uses `pytest`. Name files `test_*.py`, and place them in the matching domain folder under `tests/`. Add or update regression tests whenever behavior changes. Run targeted tests during development (for example `uv run pytest tests/rl -vv`) and run the full suite before review. There is no strict coverage threshold configured, but PRs are expected to include meaningful automated coverage. For full local runs, ensure Git LFS artifacts are present (`git lfs install && git lfs pull`).

## Commit & Pull Request Guidelines
Follow the repository’s commit style seen in history: concise imperative subjects with type/scope, such as `feat(datasets): ...`, `fix(robots): ...`, `docs: ...`, often with a PR reference like `(#2885)`. For PRs, follow `.github/PULL_REQUEST_TEMPLATE.md`: include type/scope, motivation, related issues, concrete change list, and how changes were tested. Before requesting review, run `pre-commit run -a`, run relevant `pytest` commands, and update docs for user-facing changes.
