#!/usr/bin/env bash
# Shared conda-activation helper, sourced (not executed) by the run_*.sh
# scripts in this directory. Portable across machines: finds conda via
# `conda info --base` rather than hardcoding an install path, and lets the
# env name be overridden with CONDA_ENV if the remote machine uses a
# different one than "graph-vls".
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-graph-vls}"

if ! command -v conda &> /dev/null; then
    echo "conda not found on PATH -- activate your environment manually before running this script." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Repo root: one level up from this scripts/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
