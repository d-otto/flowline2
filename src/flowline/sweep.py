import subprocess
import sys
import time
import yaml
import json
from pathlib import Path
from copy import deepcopy
import itertools
import traceback

import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
import xarray as xr

from flowline.entrypoints import run_flowline_simulation

class FlowlineSweep:
    """
    Manages the configuration, execution, and result aggregation of a
    flowline model parameter sweep.
    """
    def __init__(self, config_file, output_dir=None, workers=None, no_combine=False):
        """
        Initializes the sweep.

        Parameters
        ----------
        config_file : str or Path
            Path to the sweep YAML configuration file.
        output_dir : str or Path, optional
            Directory to save sweep results. If None, a timestamped directory
            is created.
        workers : int, optional
            Number of Dask workers (cores) to use. Defaults to all available.
        no_combine : bool, optional
            If True, do not combine individual run outputs into a single file.
        """
        self.config_path = Path(config_file)
        if output_dir is None:
            output_dir = f"sweep_output_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.no_combine = no_combine
        
        self._load_config()

    def _load_config(self):
        """Loads the YAML configuration file."""
        print(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _generate_run_params(self):
        """Generate individual run parameter dictionaries from a sweep config."""
        base_params = self.config.get('base_parameters', {})
        sweep_params = self.config.get('sweep_parameters', {})

        if not sweep_params:
            return [base_params]

        sweep_keys = list(sweep_params.keys())
        sweep_value_lists = [sweep_params[key] for key in sweep_keys]
        
        run_params_list = []
        for combination in itertools.product(*sweep_value_lists):
            run_params = deepcopy(base_params)
            for i, key in enumerate(sweep_keys):
                parts = key.split('.')
                d = run_params
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = combination[i]
            run_params_list.append(run_params)
        
        return run_params_list

    def _get_git_revision_hash(self):
        """Get the current git commit hash."""
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "Not a git repository"

    def _save_environment(self):
        """Save the pip environment to requirements.txt."""
        try:
            reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('ascii')
            with open(self.output_dir / 'requirements.txt', 'w') as f:
                f.write(reqs)
        except:
            print("Warning: Could not save pip environment.")

    def _save_reproducibility_info(self):
        """Saves configuration and environment details."""
        with open(self.output_dir / 'config.yml', 'w') as f:
            yaml.dump(self.config, f)
        
        with open(self.output_dir / 'run_info.txt', 'w') as f:
            f.write(f"git_commit: {self._get_git_revision_hash()}\n")
        
        self._save_environment()

    def _combine_results(self, successful_runs):
        """Combines individual NetCDF outputs into a single file."""
        if not successful_runs:
            print("No successful runs to combine.")
            return

        print("Combining results into a single NetCDF file...")
        sweep_params = self.config.get('sweep_parameters', {})
        sweep_dims = list(sweep_params.keys()) if sweep_params else []
        
        def preprocess_ds(ds):
            params = json.loads(ds.attrs['run_parameters'])
            coords = {}
            for dim_key in sweep_dims:
                keys = dim_key.split('.')
                val = params
                for k in keys:
                    val = val[k]
                coord_name = dim_key.replace('.', '_')
                coords[coord_name] = val
            
            new_dims = list(coords.keys())
            if new_dims:
                return ds.assign_coords(coords).expand_dims(new_dims)
            return ds

        try:
            if sweep_dims:
                concat_dims = [d.replace('.', '_') for d in sweep_dims]
                combined_ds = xr.open_mfdataset(
                    successful_runs,
                    preprocess=preprocess_ds,
                    combine='nested',
                    concat_dim=concat_dims
                )
            else: # Single run, no sweep
                ds = xr.open_dataset(successful_runs[0])
                combined_ds = preprocess_ds(ds)
            
            combined_filepath = self.output_dir / "combined_results.nc"
            combined_ds.to_netcdf(combined_filepath)
            print(f"Combined results saved to: {combined_filepath}")
        except Exception as e:
            print(f"\nCould not combine results: {e}")
            print(traceback.format_exc())
            print("Individual run files are still available in the output directory.")

    def run(self):
        """Executes the entire parameter sweep."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sweep outputs will be saved to: {self.output_dir}")
        self._save_reproducibility_info()
        
        run_params_list = self._generate_run_params()
        print(f"Generated {len(run_params_list)} parameter sets for this sweep.")

        cluster = LocalCluster(n_workers=self.workers)
        client = Client(cluster)
        print(f"Dask dashboard link: {client.dashboard_link}")
        
        tasks = []
        for i, params in enumerate(run_params_list):
            task = dask.delayed(run_flowline_simulation)((i, params, self.output_dir))
            tasks.append(task)
            
        print("Executing sweep...")
        results = []
        with tqdm(total=len(tasks), desc="Simulations", ncols=100) as pbar:
            futures = dask.compute(tasks)[0]
            for future in futures:
                results.append(future)
                pbar.update(1)

        print("Sweep complete.")
        successful_runs = [r for r in results if not str(r).startswith("ERROR")]
        success_count = len(successful_runs)
        error_count = len(results) - success_count
        
        print("\n--- Run Summary ---")
        print(f"Successful runs: {success_count}")
        print(f"Failed runs: {error_count}")
        if error_count > 0:
            print("Check *.error files in the output directory for details.")
        print("-------------------\n")

        client.close()
        cluster.close()

        if not self.no_combine and success_count > 0:
            self._combine_results(successful_runs)
