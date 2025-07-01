#!/bin/bash
# run.sh - A simple script to execute the flowline sweep.
# This script runs the sweep with 'sweep_config.yml'.
# Any additional arguments are passed to the python script.
# e.g., ./run.sh --workers 4

set -e

echo "Executing sweep with sweep_config.yml..."
python run_sweep.py sweep_config.yml "$@"
