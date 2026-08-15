# Academic Python Project Template

A modern, production-ready Python template for academic research projects with built-in support for reproducibility, documentation, testing, and publication workflows.

[![docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://USERNAME.github.io/REPO)
[![license](https://img.shields.io/github/license/Atomic-Samplers/template)](https://github.com/Atomic-Samplers/template/blob/main/LICENSE.md)
[![style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

## 📖 Table of contents

- [Quick start](#🚀-quick-start)
- [Installation options](#⚙️-installation)
- [Code quality tools](#code-quality-tools)
- [Testing](#🧪-testing)
- [Documentation](#📚-documentation)
- [Quick file structure overview](#quick-file-structure-overview)
- [License](#license)

## 🚀 Quick start

### Choose your environment manager

<details>
<summary><b>uv</b></summary>

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the template
git clone https://github.com/Atomic-Samplers/template.git
cd template

# Create environment and install dependencies
uv sync

# Two ways to run commands with uv:

# 1. Using 'uv run' prefix
uv run pytest
uv run python scripts/my_script.py
...

# 2. Activate the environment shell
source .venv/bin/activate
pytest
python scripts/my_script.py
...
```
</details>

<details>
<summary><b>conda</b></summary>

```bash
# Install miniforge if you don't have conda
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Clone the template
git clone https://github.com/Atomic-Samplers/template.git
cd template

# Create environment
conda create -n template python=3.13
conda activate template

# Install package in editable mode
pip install -e ".[dev]"

# Run things normally
pytest
python scripts/my_script.py
...
```
</details>

<details>
<summary><b>venv + pip</b></summary>

```bash
# Assuming a non-system Python installation is available
git clone https://github.com/Atomic-Samplers/template.git
cd template

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e ".[dev]"

# Run things normally
pytest
python scripts/my_script.py
...
```
</details>

## ⚙️ Installation options

Three installation extras are available:

- `dev`: Code quality tools (linters, formatters, type checkers)
- `docs`: Documentation tools (MkDocs and Material for MkDocs)
- `tests`: Testing tools (pytest and coverage)

When installing, include any combination of these extras.

```bash
# uv
uv sync --extra dev --extra docs --extra tests

# pip or conda
pip install -e ".[dev, docs, tests]"
```
## Code quality tools

<details>
<summary><b>ruff (linting)</b></summary>

```bash
# Check code with Ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Allow unsafe fixes
ruff check --fix --unsafe-fixes .
```
</details>

<details>
<summary><b>black (formatting)</b></summary>

```bash
# Format code with Black
black .

# Check without modifying
black --check .
```
</details>

<details>
<summary><b>pyright (type checking)</b></summary>

```bash
# just run pyright
pyright
```
</details>

<details>
<summary><b>docformatter (docstring formatting)</b></summary>

```bash
# format docstrings
docformatter -i -r .
```
</details>

### Pre-commit hooks (optional)

Pre-commit hooks are useful to automatically run code linters and formatters before each commit.

```bash
# install hooks
pre-commit install

# (optional) run manually on all files
pre-commit run --all-files

# after installation, hooks run automatically on git commit
git commit -m "Your commit message" # <- hooks run at this point
```

**Configured hooks:**
- **`ruff`** - Fast Python linter (checks code quality)
- **`black`** - Code formatter (Python files and Jupyter notebooks)
- **`docformatter`** - Docstring formatter
- **`blacken-docs`** - Format code in documentation
- **`trailing-whitespace`** & **`end-of-file-fixer`** - Clean up files

## 🧪 Testing

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=template --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run printing stdout during tests
pytest -s

# Run and stop after first failure
pytest -x

# Run only tests matching a keyword
pytest -k "keyword"
```

### Writing tests

Place tests in the `tests/` directory, the folder structure does not really matter but it is recommended to mirror the structure of the `src/template/` source code folder.

```python
from template.examples.example import example_function


def test_example_function():
    assert example_function().startswith("Hello from")
```

### Code coverage

Codecov allows you to track code coverage over time and identify untested parts of your codebase, you can see the coverage report [here](https://app.codecov.io/gh/Atomic-Samplers/template). To enable coverage reports with Codecov:

1. Sign up at [codecov.io](https://about.codecov.io/)
2. Give codecov access to your GitHub repository (must be public for free plan) and configure it.
3. Add the provided `CODECOV_TOKEN` from codecov to the repository secrets on GitHub:
   - Settings → Secrets and variables → Actions
   - New repository secret: `CODECOV_TOKEN`

The workflow in `.github/workflows/tests.yml` is already set up to upload coverage reports to Codecov when pushing to the repository. Coverage uploads automatically when configured.

## 📚 Documentation

Documentation is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

### Building documentation locally

Follow the instruction to install the `docs` extra, then run:

```bash
# Serve docs locally with live reload, will open at http://localhost:8000
mkdocs serve

# Build static documentation site
mkdocs build
```

- Documentation files live in `docs/`, structured in Markdown.
- Configuration in `mkdocs.yml`

### Automatic site deployment

Using github actions, you can automatically deploy documentation to GitHub Pages on each push to `main`. Documentation automatically deploys to GitHub Pages on push to `main`. View at:
`https://USERNAME.github.io/REPO`

To enable, go to your GitHub repository:
1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` keep the folder to `root/`
4. Save

## Quick command reference

- [`uv` cheatsheet](https://docs.astral.sh/uv/getting-started/features/#python-versions)
- [`conda` cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [`pip` cheatsheet](https://pip.pypa.io/en/stable/user_guide/#cheat-sheet)

## Quick file structure overview

- `src/` - Main package source code
- `tests/` - Unit and integration tests
- `docs/` - Documentation source files
- `.venv/` - Virtual environment (if using venv)
- `.github/workflows/` - GitHub Actions workflows for CI/CD, testing, and deployment
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `mkdocs.yml` - MkDocs configuration file
- `pyproject.toml` - Project metadata and dependencies
- `src/template/py.typed` - Marker file indicating package is typed
- `.python-version` - Specifies Python version for pyenv and similar tools
- `README.md` - Project overview and instructions
- `CHANGELOG.md` - Record of changes and versions
- `CODE_OF_CONDUCT.md` - Community code of conduct
- `MANIFEST.in` - Specifies additional files to include in the package
- `LICENSE.md` - License information
- `.gitignore` - Specifies files and directories to ignore in Git

## License

This project is licensed under the terms of the BSD 3-Clause license.

## Acknowledgements

This template was heavily inspired by Quacc's project template: [https://github.com/Quacc/Quacc](https://github.com/Quacc/Quacc)
