#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --locked python -m src.data
uv run --locked python -m src.train
uv run --locked python -m src.evaluate
uv run --locked python -m src.plots
