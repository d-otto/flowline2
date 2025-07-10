"""
run.py

Example of running a parameter sweep with a spin-up stage.

This example demonstrates a common use case:
1. A steady-state "spin-up" run is performed for each parameter set to generate
   a stable initial glacier geometry.
2. A second "main" or "experiment" run is then initialized from the result of
   the spin-up run, with a different set of forcing conditions (e.g., a
   climate warming scenario).
"""
import argparse
from pathlib import Path
import sys

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep

def main():
    """
    Sets up and runs the flowline model parameter sweep.
    """
    parser = argparse.ArgumentParser(
        description="Run a flowline model parameter sweep with a spin-up stage."
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(Path(__file__).resolve().parent / 'config.yml'),
        help="Path to the sweep configuration YAML file."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'output'),
        help="Directory to save sweep results."
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Number of Dask workers (processes) to use. Defaults to all available cores."
    )
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help="If set, do not combine individual run outputs into a single NetCDF file."
    )
    args = parser.parse_args()

    # Initialize and run the sweep
    sweep = FlowlineSweep(
        config_file=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        no_combine=args.no_combine
    )
    sweep.run()

if __name__ == '__main__':
    main()
