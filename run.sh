#!/bin/bash
# run.sh - A simple script to execute the flowline sweep.
# This script runs the sweep with 'sweep_config.yml'.
# Any additional arguments are passed to the flowline-sweep command.
# e.g., ./run.sh --workers 4
#
# Assumes you have installed the package, e.g., with 'pip install -e .'

set -e

echo "Executing sweep with sweep_config.yml..."
flowline-sweep sweep_config.yml "$@"
