import argparse
import subprocess
from pathlib import Path
import yaml
import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
import time
import xarray as xr
import sys
import json

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.flowline.io import generate_run_params
from src.flowline.entrypoints import run_flowline_simulation

def get_git_revision_hash():
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "Not a git repository"

def save_environment(output_dir):
    """Save the pip environment to requirements.txt."""
    try:
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('ascii')
        with open(output_dir / 'requirements.txt', 'w') as f:
            f.write(reqs)
    except:
        print("Warning: Could not save pip environment.")

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

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep outputs will be saved to: {output_dir}")

    # Load and copy config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)

    # Save reproducibility info
    with open(output_dir / 'run_info.txt', 'w') as f:
        f.write(f"git_commit: {get_git_revision_hash()}\n")
    save_environment(output_dir)

    # Generate parameter sets
    run_params_list = list(generate_run_params(config))
    print(f"Generated {len(run_params_list)} parameter sets for this sweep.")

    # Setup Dask
    cluster = LocalCluster(n_workers=args.workers)
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    # Create Dask tasks
    tasks = []
    for i, params in enumerate(run_params_list):
        task = dask.delayed(run_flowline_simulation)((i, params, output_dir))
        tasks.append(task)

    # Run tasks with a progress bar
    print("Executing sweep...")
    results = []
    with tqdm(total=len(tasks), desc="Simulations", ncols=100) as pbar:
        futures = dask.compute(tasks)[0]
        for future in futures:
            results.append(future)
            pbar.update(1)

    print("Sweep complete.")
    print("\n--- Run Summary ---")
    success_count = sum(1 for r in results if not str(r).startswith("ERROR"))
    error_count = len(results) - success_count
    print(f"Successful runs: {success_count}")
    print(f"Failed runs: {error_count}")
    if error_count > 0:
        print("Check *.error files in the output directory for details.")
    print("-------------------\n")

    client.close()
    cluster.close()
    
    # Combine results into a single file
    if not args.no_combine and success_count > 0:
        print("Combining results into a single NetCDF file...")
        sweep_dims = list(config.get('sweep_parameters', {}).keys())
        
        def preprocess_ds(ds):
            params = json.loads(ds.attrs['run_parameters'])
            coords = {}
            for dim_key in sweep_dims:
                keys = dim_key.split('.')
                val = params
                for k in keys:
                    val = val[k]
                # Convert list to string for use as a coordinate
                if isinstance(val, list):
                    val = str(val)
                coord_name = dim_key.replace('.', '_')
                coords[coord_name] = val
            return ds.assign_coords(coords).expand_dims(list(coords.keys()))

        output_files = [r for r in results if not str(r).startswith("ERROR")]
        
        try:
            # Determine correct dimension order for concat
            concat_dims = [d.replace('.', '_') for d in sweep_dims]

            combined_ds = xr.open_mfdataset(
                output_files,
                preprocess=preprocess_ds,
                combine='nested',
                concat_dim=concat_dims
            )
            
            combined_filepath = output_dir / "combined_results.nc"
            combined_ds.to_netcdf(combined_filepath)
            print(f"Combined results saved to: {combined_filepath}")
        except Exception as e:
            print(f"\nCould not combine results: {e}")
            print("Individual run files are still available in the output directory.")


if __name__ == "__main__":
    main()
