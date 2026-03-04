#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/schedule_reports.sh
# Refreshes local daily/weekly reports in outputs/reports.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMEZONE="${NSB_TIMEZONE:-local}"

export PYTHONPATH=src

$PYTHON_BIN -m noslouchbench.cli summarize-habits --timezone "$TIMEZONE"
