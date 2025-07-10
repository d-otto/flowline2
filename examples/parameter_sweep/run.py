import argparse
from pathlib import Path
import sys

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.flowline.sweep import FlowlineSweep

def main():
    parser = argparse.ArgumentParser(description="Run a flowline model parameter sweep.")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / 'config.yml'),
                        help="Path to the sweep YAML configuration file.")
    parser.add_argument("-o", "--output_dir", type=str,
                        default=str(Path(__file__).resolve().parent / 'output'),
                        help="Directory to save sweep results.")
    parser.add_argument("--workers", type=int, default=None, help="Number of Dask workers (cores) to use. Defaults to all available.")
    parser.add_argument("--no-combine", action="store_true", help="Do not combine individual run outputs into a single file after the sweep.")
    args = parser.parse_args()

    sweep = FlowlineSweep(
        config_file=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        no_combine=args.no_combine
    )
    sweep.run()


if __name__ == "__main__":
    main()
