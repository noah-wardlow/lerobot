# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is Hugging Face's robotics library providing models, datasets, and tools for real-world robotics in PyTorch. It includes hardware-agnostic robot control, a standardized dataset format (LeRobotDataset: Parquet + MP4), and state-of-the-art policy implementations (ACT, Diffusion, TDMPC, SmolVLA, VLA models, etc.).

## Common Commands

### Installation (development)
```bash
pip install -e ".[dev,test]"       # Basic dev install
pip install -e ".[all]"            # Full install with all extras
pre-commit install                 # Set up pre-commit hooks
git lfs install && git lfs pull    # Required for test artifacts
```

### Testing
```bash
pytest tests -vv --maxfail=10                  # Full test suite (mirrors CI)
pytest -sv tests/test_specific.py              # Single test file
pytest -sv tests/test_specific.py::test_name   # Single test function
make test-end-to-end DEVICE=cpu                # E2E tests (ACT, Diffusion, TDMPC, SmolVLA)
```

### Linting & Formatting
```bash
pre-commit run --all-files    # Run all checks (ruff, mypy, typos, bandit, etc.)
ruff check src/               # Lint only
ruff format src/               # Format only
```

### Key CLI Entry Points
```bash
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
lerobot-eval --policy.path=outputs/train/.../pretrained_model
lerobot-record, lerobot-teleoperate, lerobot-replay, lerobot-info
```

## Code Architecture

### Source Layout
All source code lives under `src/lerobot/`. The package uses `setuptools` with `[tool.setuptools.packages.find] where = ["src"]`.

### Configuration System
Configs are **dataclass-based** using the `draccus` library (pinned at 0.10.0). The main training entrypoint parses `TrainPipelineConfig` (in `configs/train.py`) which composes:
- `PreTrainedConfig` (policy config, in `configs/policies/`)
- `DatasetConfig`, `EvalConfig`, `WandBConfig` (in `configs/default.py`)
- `OptimizerConfig`, `LRSchedulerConfig` (in `optim/`)
- `EnvConfig` (in `envs/configs.py`)

CLI arguments override config fields via draccus (e.g., `--policy.type=act --batch_size=32`). Policies can also define training presets for optimizer/scheduler via `get_optimizer_preset()` / `get_scheduler_preset()`.

### Policy Architecture
Each policy lives in `policies/<name>/` with a consistent structure:
- `configuration_<name>.py` — Config dataclass (extends `PreTrainedConfig`)
- `modeling_<name>.py` — Model class (extends `PreTrainedPolicy`)
- `processor_<name>.py` — Pre/post-processor factory (normalization, tokenization, etc.)

Policies are instantiated through factories in `policies/factory.py`:
- `get_policy_class(name)` — Lazily imports the policy class by name
- `make_policy(cfg, ds_meta, env_cfg)` — Full construction with feature inference
- `make_pre_post_processors(policy_cfg)` — Creates `PolicyProcessorPipeline` instances

The `PolicyProcessorPipeline` (in `processor/`) is a composable chain of data transforms applied during training and inference.

### Hardware Abstraction
Robots, cameras, motors, and teleoperators each define a base interface:
- `robots/` — `Robot` base class: `connect()`, `get_observation()`, `send_action()`
- `motors/` — Bus drivers: Dynamixel, Feetech, Damiao
- `cameras/` — Camera implementations with a common interface
- `teleoperators/` — Teleoperation device interfaces

### Dataset Format
`LeRobotDataset` (in `datasets/lerobot_dataset.py`) standardizes robotic data as Parquet files + MP4 videos (or images), with HuggingFace Hub integration for storage and streaming.

### Scripts
CLI entry points are in `scripts/` and registered in `pyproject.toml [project.scripts]`. The main ones are `lerobot_train.py` and `lerobot_eval.py`.

## Code Style

- **Line length**: 110 characters
- **Formatter/Linter**: ruff (target Python 3.10)
- **Quote style**: double quotes
- **Import sorting**: isort via ruff, with `known-first-party = ["lerobot"]`
- **Type checking**: mypy is enabled strictly for `configs/`, `optim/`, `model/`, `cameras/`, `transport/`, `envs/` modules; other modules have `ignore_errors = true` (gradual adoption)
- Generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from linting

## Dependency Notes

- Python >=3.10 required
- Many features are behind optional extras (e.g., `pip install -e ".[smolvla,aloha]"`)
- `wallx` and `pi` extras have `transformers` version conflicts with other extras — these are declared in `[tool.uv] conflicts` in pyproject.toml
- `uv` is the preferred package manager (used in CI)
