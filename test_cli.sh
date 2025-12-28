#!/bin/bash
# Test script for DeepFense CLI without installation
# Usage: ./test_cli.sh [command] [args...]

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -m deepfense.cli.main "$@"

