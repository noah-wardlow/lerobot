# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- Run all tests: `python -m pytest tests/`
- Run specific test file: `python -m pytest tests/path/to/test_file.py`
- Run tests with coverage: `python -m pytest tests/ --cov=src/lerobot --cov-report=term-missing`
- Run end-to-end tests: `make test-end-to-end` (requires specific env setup)

### Code Quality
- Format code: `ruff format .`
- Lint code: `ruff check . --fix`
- Run security checks: `bandit -c pyproject.toml -r src/`
- Run all pre-commit hooks: `pre-commit run --all-files`
- Check types: `mypy src/lerobot` (when enabled)

### Training & Evaluation
- Train a policy: `python -m lerobot.scripts.train --config_path=path/to/config`
- Evaluate a policy: `python -m lerobot.scripts.eval --policy.path=path/to/model`
- Visualize dataset: `python -m lerobot.scripts.visualize_dataset --repo-id=lerobot/dataset_name`

### Build & Installation
- Install in development mode: `pip install -e .`
- Install with specific robot extras: `pip install -e ".[aloha,pusht]"`
- Build Docker images: `make build-user` or `make build-internal`

## Architecture Overview

LeRobot is a PyTorch-based robotics library with a modular architecture:

### Core Structure
- **`src/lerobot/`**: Main package source code
- **`lerobot/policies/`**: Policy implementations (ACT, Diffusion, TDMPC, Pi0, SmolVLA, etc.)
- **`lerobot/datasets/`**: Dataset handling, transforms, and utilities
- **`lerobot/robots/`**: Robot-specific implementations and configurations
- **`lerobot/cameras/`**: Camera integrations (OpenCV, RealSense)
- **`lerobot/motors/`**: Motor control (Dynamixel, Feetech)
- **`lerobot/scripts/`**: Command-line tools for training, evaluation, visualization

### Key Components

#### Policies
Each policy (e.g., ACT, Diffusion) has:
- `configuration_*.py`: Dataclass config defining hyperparameters
- `modeling_*.py`: PyTorch model implementation
- Integration through `lerobot/policies/factory.py`

#### Datasets
- `LeRobotDataset`: Core dataset class supporting temporal indexing
- Support for video compression, image transforms, and HuggingFace Hub integration
- Multi-modal data (cameras, robot states, actions) with flexible serialization

#### Robots & Hardware
- Modular robot implementations in `lerobot/robots/`
- Each robot has configuration files and implementation classes
- Support for various hardware: SO-100/101, Koch, HopeJR, LeKiwi, etc.
- Motor control abstraction supporting different servo types

### Configuration System
- Uses `draccus` for configuration management
- Config files define training hyperparameters, model architecture, dataset settings
- Supports config inheritance and command-line overrides
- Main configs in `lerobot/configs/`

### Data Format
- Custom LeRobotDataset format with metadata, statistics, episode indexing
- Video storage for efficiency with temporal relationship querying
- Integration with HuggingFace datasets and hub for sharing

## Development Notes

- Python 3.10+ required
- Uses `ruff` for formatting and linting (configuration in `pyproject.toml`)
- Pre-commit hooks enforce code quality
- Extensive test suite covering policies, datasets, hardware integration
- Docker support for consistent environments
- Weights & Biases integration for experiment tracking
- Real-world robot support with teleoperation and data collection tools