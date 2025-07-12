import subprocess
import sys
import time
import json
from pathlib import Path
from copy import deepcopy
import itertools
import traceback
import math

import dask
import dask.delayed
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
import xarray as xr

from flowline.entrypoints import run_flowline_simulation, run_spinup_simulation
from flowline.visualization import plot_sweep_qc

class FlowlineSweep:
    """
    Manages the configuration, execution, and result aggregation of a
    flowline model parameter sweep.
    """
    def __init__(self, base_config, base_geometry, base_forcing, sweep_parameters, 
                 spinup_config=None, output_dir=None, workers=None, no_combine=False, no_progress=False):
        """
        Initializes the sweep.

        Parameters
        ----------
        base_config : FlowlineConfig
            Base configuration object for all runs.
        base_geometry : FlowlineGeometry
            Base geometry object for all runs.
        base_forcing : MassBalanceForcing
            Base forcing object for all runs.
        sweep_parameters : dict
            Dictionary mapping parameter names to lists of values to sweep over.
            Parameter names should be in the format 'object.attribute' (e.g., 'config.tf', 'forcing.T0').
        spinup_config : dict, optional
            Spinup configuration with mode-based control. Format:
            {
                'mode': 'shared|individual|per_run_custom|from_file',
                'enabled': True,
                'config': {...},    # Config overrides (tf, delt, etc.)
                'forcing': {...},   # REQUIRED: Explicit forcing parameters
                'geometry': {...},  # Geometry overrides (optional)
                'profile_path': '...',  # For 'from_file' mode
                'customizations': [     # For 'per_run_custom' mode
                    {'run_ids': ['run_0001', 'run_0002'], 'forcing': {'T0': 8.0}},
                    {'run_ids': ['run_0003'], 'forcing': {'T0': 9.0}}
                ]
            }
            NOTE: Explicit forcing parameters are always required in 'forcing' 
            to ensure compatibility with spinup timespan and configuration.
        output_dir : str or Path, optional
            Directory to save sweep results. If None, a timestamped directory
            is created.
        workers : int, optional
            Number of Dask workers (cores) to use. Defaults to all available.
        no_combine : bool, optional
            If True, do not combine individual run outputs into a single file.
        no_progress : bool, optional
            If True, disable progress bars (tqdm).
        """
        self.base_config = base_config
        self.base_geometry = base_geometry
        self.base_forcing = base_forcing
        self.sweep_parameters = sweep_parameters
        self.spinup_config = spinup_config or {}
        
        if output_dir is None:
            output_dir = f"sweep_output_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.no_combine = no_combine
        self.no_progress = no_progress


    def _generate_run_objects(self):
        """Generate individual run objects from sweep parameters."""
        if not self.sweep_parameters:
            return [(self.base_config, self.base_geometry, self.base_forcing)]

        sweep_keys = list(self.sweep_parameters.keys())
        sweep_value_lists = [self.sweep_parameters[key] for key in sweep_keys]
        
        run_objects_list = []
        for combination in itertools.product(*sweep_value_lists):
            # Create copies of base objects
            config = deepcopy(self.base_config)
            geometry = deepcopy(self.base_geometry)
            forcing = deepcopy(self.base_forcing)
            
            # Apply sweep parameter values
            for i, key in enumerate(sweep_keys):
                parts = key.split('.')
                if parts[0] == 'config':
                    setattr(config, parts[1], combination[i])
                elif parts[0] == 'geometry':
                    setattr(geometry, parts[1], combination[i])
                elif parts[0] == 'forcing':
                    setattr(forcing, parts[1], combination[i])
                else:
                    raise ValueError(f"Unknown object type in sweep parameter: {parts[0]}")
            
            run_objects_list.append((config, geometry, forcing))
        
        return run_objects_list

    def _get_run_id(self, run_index):
        """Generate run ID for a given run index."""
        return f"run_{run_index:04d}"

    def _orchestrate_spinups(self, run_objects_list, client):
        """
        Orchestrate spinup runs based on spinup_config mode.
        Returns a mapping of run_id -> profile_path.
        """
        if not self.spinup_config or not self.spinup_config.get('enabled', False):
            print("No spinup configuration - skipping spinup phase.")
            return {}
        
        spinup_mode = self.spinup_config.get('mode', 'individual')
        print(f"Spinup mode: {spinup_mode}")
        
        if spinup_mode == 'from_file':
            return self._handle_file_spinup(run_objects_list)
        elif spinup_mode == 'shared':
            return self._handle_shared_spinup(run_objects_list, client)
        elif spinup_mode == 'per_run_custom':
            return self._handle_custom_spinups(run_objects_list, client)
        elif spinup_mode == 'individual':
            print("Individual spinup mode - each experimental run will handle its own spinup.")
            return self._handle_individual_spinups(run_objects_list, client)
        else:
            raise ValueError(f"Unknown spinup mode: {spinup_mode}")

    def _handle_file_spinup(self, run_objects_list):
        """Handle 'from_file' spinup mode."""
        profile_path = self.spinup_config.get('profile_path')
        if not profile_path:
            raise ValueError("profile_path is required for 'from_file' spinup mode")
        
        if not Path(profile_path).exists():
            raise FileNotFoundError(f"Spinup profile file not found: {profile_path}")
        
        print(f"Using existing spinup profile: {profile_path}")
        
        # All runs use the same profile
        run_profile_mapping = {}
        for i in range(len(run_objects_list)):
            run_id = self._get_run_id(i)
            run_profile_mapping[run_id] = str(profile_path)
        
        return run_profile_mapping

    def _handle_shared_spinup(self, run_objects_list, client):
        """Handle 'shared' spinup mode - single spinup for all runs."""
        print("Running single shared spinup...")
        
        # Use base objects for shared spinup
        spinup_config = self._create_spinup_config(self.base_config)
        spinup_geometry = self._create_spinup_geometry(self.base_geometry)
        spinup_forcing = self._create_spinup_forcing(self.base_forcing, spinup_config=spinup_config)
        
        # Run single spinup
        spinup_task = dask.delayed(run_spinup_simulation)(
            ("shared", spinup_config, spinup_geometry, spinup_forcing, 
             self.output_dir, self.no_progress)
        )
        
        print("Executing shared spinup...")
        spinup_result = dask.compute(spinup_task)[0]
        
        if str(spinup_result).startswith("ERROR"):
            raise RuntimeError(f"Shared spinup failed: {spinup_result}")
        
        print(f"Shared spinup completed: {spinup_result}")
        
        # All runs use the same profile
        run_profile_mapping = {}
        for i in range(len(run_objects_list)):
            run_id = self._get_run_id(i)
            run_profile_mapping[run_id] = spinup_result
        
        return run_profile_mapping

    def _handle_custom_spinups(self, run_objects_list, client):
        """Handle 'per_run_custom' spinup mode."""
        print("Running customized spinups...")
        
        customizations = self.spinup_config.get('customizations', [])
        spinup_tasks = []
        run_spinup_map = {}  # run_id -> spinup_task
        
        # Build customization mapping: run_id -> custom_params
        custom_params_map = {}
        for custom in customizations:
            run_ids = custom.get('run_ids', [])
            custom_overrides = {k: v for k, v in custom.items() if k != 'run_ids'}
            for run_id in run_ids:
                custom_params_map[run_id] = custom_overrides
        
        # Create spinup tasks for each unique customization
        unique_customs = {}  # custom_params_hash -> (task, custom_params)
        
        for i, (config, geometry, forcing) in enumerate(run_objects_list):
            run_id = self._get_run_id(i)
            custom_params = custom_params_map.get(run_id, {})
            
            # Hash custom parameters to identify unique spinups
            custom_hash = hash(str(sorted(custom_params.items())))
            
            if custom_hash not in unique_customs:
                # Create customized spinup objects
                spinup_config = self._create_spinup_config(config, custom_params.get('config', {}))
                spinup_geometry = self._create_spinup_geometry(geometry, custom_params.get('geometry', {}))
                spinup_forcing = self._create_spinup_forcing(forcing, custom_params, spinup_config=spinup_config)
                
                spinup_task = dask.delayed(run_spinup_simulation)(
                    (f"custom_{custom_hash}", spinup_config, spinup_geometry, spinup_forcing,
                     self.output_dir, self.no_progress)
                )
                
                unique_customs[custom_hash] = (spinup_task, custom_params)
            
            run_spinup_map[run_id] = custom_hash
        
        # Execute all unique spinup tasks
        spinup_tasks = [task for task, _ in unique_customs.values()]
        print(f"Executing {len(spinup_tasks)} unique spinup configurations...")
        
        if self.no_progress:
            spinup_results = dask.compute(spinup_tasks)[0]
        else:
            with tqdm(total=len(spinup_tasks), desc="Spinup runs", ncols=100) as pbar:
                spinup_results = dask.compute(spinup_tasks)[0]
                for _ in spinup_results:
                    pbar.update(1)
        
        # Build results mapping: custom_hash -> profile_path
        custom_results = {}
        for i, (custom_hash, (task, custom_params)) in enumerate(unique_customs.items()):
            result = spinup_results[i]
            if str(result).startswith("ERROR"):
                raise RuntimeError(f"Custom spinup failed for {custom_params}: {result}")
            custom_results[custom_hash] = result
        
        # Map run_ids to their profile paths
        run_profile_mapping = {}
        for run_id, custom_hash in run_spinup_map.items():
            run_profile_mapping[run_id] = custom_results[custom_hash]
        
        print(f"Custom spinups completed. {len(run_profile_mapping)} profiles assigned.")
        return run_profile_mapping

    def _handle_individual_spinups(self, run_objects_list, client):
        """Handle 'individual' spinup mode - each run gets its own spinup."""
        print("Running individual spinups...")
        
        spinup_tasks = []
        for i, (config, geometry, forcing) in enumerate(run_objects_list):
            run_id = self._get_run_id(i)
            
            # Create customized spinup objects for this run
            spinup_config = self._create_spinup_config(config)
            spinup_geometry = self._create_spinup_geometry(geometry)
            spinup_forcing = self._create_spinup_forcing(forcing, spinup_config=spinup_config)
            
            spinup_task = dask.delayed(run_spinup_simulation)(
                (run_id, spinup_config, spinup_geometry, spinup_forcing,
                 self.output_dir, self.no_progress)
            )
            spinup_tasks.append(spinup_task)
        
        # Execute all spinup tasks
        print(f"Executing {len(spinup_tasks)} individual spinups...")
        
        if self.no_progress:
            spinup_results = dask.compute(spinup_tasks)[0]
        else:
            with tqdm(total=len(spinup_tasks), desc="Individual spinups", ncols=100) as pbar:
                spinup_results = dask.compute(spinup_tasks)[0]
                for _ in spinup_results:
                    pbar.update(1)
        
        # Build run_id -> profile_path mapping
        run_profile_mapping = {}
        for i, result in enumerate(spinup_results):
            run_id = self._get_run_id(i)
            if str(result).startswith("ERROR"):
                raise RuntimeError(f"Individual spinup failed for {run_id}: {result}")
            run_profile_mapping[run_id] = result
        
        print(f"Individual spinups completed. {len(run_profile_mapping)} profiles created.")
        return run_profile_mapping

    def _create_spinup_config(self, base_config, custom_overrides=None):
        """Create spinup config by applying base spinup config and custom overrides."""
        spinup_config = deepcopy(base_config)
        
        # Apply base spinup config overrides
        base_overrides = self.spinup_config.get('config', {})
        for key, value in base_overrides.items():
            setattr(spinup_config, key, value)
        
        # Apply custom overrides if provided
        if custom_overrides:
            for key, value in custom_overrides.items():
                setattr(spinup_config, key, value)
        
        return spinup_config

    def _create_spinup_geometry(self, base_geometry, custom_overrides=None):
        """Create spinup geometry by applying base spinup config and custom overrides."""
        spinup_geometry = deepcopy(base_geometry)
        
        # Remove any existing profile to start from h_init
        if hasattr(spinup_geometry, 'profile'):
            spinup_geometry.profile = None
        
        # Apply base spinup geometry overrides
        base_overrides = self.spinup_config.get('geometry', {})
        for key, value in base_overrides.items():
            setattr(spinup_geometry, key, value)
        
        # Apply custom overrides if provided
        if custom_overrides:
            for key, value in custom_overrides.items():
                setattr(spinup_geometry, key, value)
        
        return spinup_geometry

    def _create_spinup_forcing(self, base_forcing, custom_overrides=None, spinup_config=None):
        """Create spinup forcing by applying base spinup config and custom overrides."""
        from flowline.flowline2d import TemperaturePrecipitationForcing, DirectMassBalanceForcing
        
        # Get all overrides (base + custom)
        base_overrides = self.spinup_config.get('forcing', {})
        all_overrides = base_overrides.copy()
        if custom_overrides:
            all_overrides.update(custom_overrides)
        
        # Always require explicit spinup forcing
        if not all_overrides:
            raise ValueError(
                "Explicit spinup forcing parameters are required in spinup_config['forcing']. "
                "This ensures spinup forcing is compatible with your spinup timespan and configuration.\n"
                "Example: spinup_config = {\n"
                "    'config': {'tf': 500},\n"
                "    'forcing': {'T0': 8.0, 'P0': 2.0}  # Constant forcing for spinup\n"
                "}"
            )
        
        # Check if we have a spinup config with timing info for forcing creation
        if spinup_config and hasattr(spinup_config, 'tf'):
            # For TemperaturePrecipitationForcing, create new instance with proper timespan
            if isinstance(base_forcing, TemperaturePrecipitationForcing):
                forcing_params = {
                    'T0': all_overrides.get('T0', 8.0),
                    'P0': all_overrides.get('P0', 2.0),
                    'gamma': all_overrides.get('gamma', 6.5e-3),
                    'mu': all_overrides.get('mu', 0.65),
                    'tf': spinup_config.tf
                }
                # Include any other forcing parameters from overrides
                for key, value in all_overrides.items():
                    if key not in forcing_params:
                        forcing_params[key] = value
                
                return TemperaturePrecipitationForcing(**forcing_params)
        
        # For other cases, copy base forcing and apply overrides
        spinup_forcing = deepcopy(base_forcing)
        for key, value in all_overrides.items():
            setattr(spinup_forcing, key, value)
        
        return spinup_forcing

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
        # Save sweep parameters and object info
        config_info = {
            'sweep_parameters': self.sweep_parameters,
            'spinup_config': self.spinup_config,
            'base_config': self.base_config.__dict__ if hasattr(self.base_config, '__dict__') else str(self.base_config),
            'base_geometry': {
                'x_gr_shape': self.base_geometry.x_gr.shape if hasattr(self.base_geometry, 'x_gr') else None,
                'geometry_type': type(self.base_geometry).__name__
            },
            'base_forcing': {
                'forcing_type': type(self.base_forcing).__name__,
                'forcing_attrs': {k: v for k, v in self.base_forcing.__dict__.items() if not k.startswith('_')}
            }
        }
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_info, f, indent=2, default=str)
        
        with open(self.output_dir / 'run_info.txt', 'w') as f:
            f.write(f"git_commit: {self._get_git_revision_hash()}\n")
        
        self._save_environment()

    def _combine_results(self, successful_runs):
        """Combines individual NetCDF outputs into a single file."""
        if not successful_runs:
            print("No successful runs to combine.")
            return

        print("Combining results into a single NetCDF file...")
        sweep_dims = list(self.sweep_parameters.keys()) if self.sweep_parameters else []
        
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
                # Sort runs to ensure a predictable order for xarray's nested combine.
                # Filenames include a zero-padded index, so alphabetical sort works.
                sorted_runs = sorted(successful_runs)
                concat_dims = [d.replace('.', '_') for d in sweep_dims]

                # For nested combine, xarray expects a nested list of files that
                # matches the structure of the swept dimensions.
                sweep_value_lists = [self.sweep_parameters[key] for key in sweep_dims]
                shape = [len(v) for v in sweep_value_lists]

                def _nest_list(flat_list, shape_dims):
                    """Recursively nest a flat list to a given shape."""
                    if not shape_dims:
                        return flat_list[0]
                    n = shape_dims[0]
                    chunk_size = len(flat_list) // n
                    return [_nest_list(flat_list[i*chunk_size:(i+1)*chunk_size], shape_dims[1:]) for i in range(n)]

                # Only nest if there is more than one dimension to sweep over.
                # If some runs failed, we can't reshape into a hyper-rectangle,
                # so we fall back to a 1D concatenation and warn the user.
                expected_runs = math.prod(shape) if shape else 0
                if len(shape) > 1 and len(sorted_runs) == expected_runs:
                    runs_for_xr = _nest_list(sorted_runs, shape)
                else:
                    if len(shape) > 1 and len(sorted_runs) != expected_runs:
                        print(f"\nWarning: Number of successful runs ({len(sorted_runs)}) does not match expected "
                              f"from sweep parameters ({expected_runs}). Combining as a 1D list, which may "
                              "produce incorrect dimensions or fail if sweep dimensions are not orthogonal.")
                    runs_for_xr = sorted_runs

                combined_ds = xr.open_mfdataset(
                    runs_for_xr,
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

            # Generate and save QC plots for the sweep
            plot_sweep_qc(combined_ds, self.output_dir)
        except Exception as e:
            print(f"\nCould not combine results: {e}")
            print(traceback.format_exc())
            print("Individual run files are still available in the output directory.")

    def run(self):
        """Executes the entire parameter sweep."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sweep outputs will be saved to: {self.output_dir}")
        self._save_reproducibility_info()
        
        run_objects_list = self._generate_run_objects()
        print(f"Generated {len(run_objects_list)} object sets for this sweep.")

        cluster = LocalCluster(n_workers=self.workers)
        client = Client(cluster)
        print(f"Dask dashboard link: {client.dashboard_link}")
        
        # --- Spinup Phase ---
        run_profile_mapping = self._orchestrate_spinups(run_objects_list, client)
        
        # --- Experimental Phase ---
        exp_tasks = []
        for i, (config, geometry, forcing) in enumerate(run_objects_list):
            run_id = self._get_run_id(i)
            
            # Apply assigned spinup profile to geometry
            if run_id in run_profile_mapping:
                geometry = deepcopy(geometry)  # Don't modify original
                geometry.profile = run_profile_mapping[run_id]
                if hasattr(geometry, 'h_init'):
                    geometry.h_init = None
            
            task = dask.delayed(run_flowline_simulation)((run_id, config, geometry, forcing, 
                                                          self.output_dir, self.no_progress))
            exp_tasks.append(task)
            
        print("Executing experimental runs...")
        results = []
        if self.no_progress:
            futures = dask.compute(exp_tasks)[0]
            for future in futures:
                results.append(future)
        else:
            with tqdm(total=len(exp_tasks), desc="Experimental runs", ncols=100) as pbar:
                futures = dask.compute(exp_tasks)[0]
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
