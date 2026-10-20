#!/usr/bin/env bash
set -euo pipefail
"${RL_UV_BIN:-uv}" pip compile pyproject.toml --python-version 3.13 --extra yahoo --extra alpaca --extra dev --extra trees --generate-hashes --output-file requirements.lock
