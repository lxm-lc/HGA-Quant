#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. The implementation lives in the explicit
# HGA-Quant wrapper so both names run the same single-task workflow.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_hga_flexvar.sh" "$@"
