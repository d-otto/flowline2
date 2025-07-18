import subprocess
import sys
import time
import json
from pathlib import Path
from copy import deepcopy
import itertools
import traceback
import math
from typing import Dict, List, Any, Optional, Union, Tuple

import dask
import dask.delayed
from dask.base import compute
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
import xarray as xr

from flowline.entrypoints import run_flowline_simulation, run_spinup_simulation
from flowline.visualization import plot_sweep_qc
from flowline.spinup import FlowlineSpinup

class FlowlineSweep:
    """
    Manages the configuration, execution, and result aggregation of a
    flowline model parameter sweep.
    """
    def __init__(self, base_config: Any, base_geometry: Any, base_forcing: Any, 
                 sweep_parameters: Optional[Dict[str, List[Any]]] = None, 
                 spinup_config: Optional[Dict[str, Any]] = None, 
                 spinup_objects: Optional[Union[Any, Dict[str, Any]]] = None, 
                 experimental_perturbations: Optional[Dict[str, Dict[str, Any]]] = None, 
                 output_dir: Optional[Union[str, Path]] = None, 
                 workers: Optional[int] = None, no_combine: bool = False, no_progress: bool = False):
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
        sweep_parameters : dict, optional
            Dictionary mapping parameter names to lists of values to sweep over.
            Parameter names should be in the format 'object.attribute' (e.g., 'config.tf', 'forcing.T0').
        spinup_config : dict, optional
            Legacy spinup configuration with mode-based control. See documentation
            for detailed format. Cannot be used together with spinup_objects.
        spinup_objects : FlowlineSpinup or dict, optional  
            FlowlineSpinup object(s) for new 4-object architecture. Can be:
            - Single FlowlineSpinup object: used for all runs
            - Dict mapping run_id -> FlowlineSpinup object: specific spinup per run
            Example: spinup_obj or {'run_0001': spinup_obj1, 'run_0002': spinup_obj2}
            Cannot be used together with spinup_config.
        experimental_perturbations : dict, optional
            Mapping of run_id -> perturbations dict for applying experimental changes after spinup.
            Perturbations use lambda functions for relative changes:
            Example: {'run_0001': {'forcing.T0': lambda T0: T0 + 1.0, 'config.tf': lambda _: 200}}
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
        self.spinup_objects = spinup_objects
        self.experimental_perturbations = experimental_perturbations or {}
        
        # Validation: cannot use both spinup approaches
        if self.spinup_config and self.spinup_objects:
            raise ValueError("Cannot specify both spinup_config and spinup_objects. "
                           "Use spinup_config for legacy mode-based control or "
                           "spinup_objects for the new 4-object architecture.")
        
        if output_dir is None:
            output_dir = f"sweep_output_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.no_combine = no_combine
        self.no_progress = no_progress


    def _generate_run_objects(self) -> List[Tuple[Any, Any, Any]]:
        """Generate individual run objects from sweep parameters or spinup_objects."""
        
        # If using spinup_objects with no sweep_parameters, infer runs from spinup_objects
        if self.spinup_objects and not self.sweep_parameters:
            if isinstance(self.spinup_objects, dict):
                # Dict format: create one run per spinup object
                run_count = len(self.spinup_objects)
            else:
                # Single spinup object: need to determine run count from experimental_perturbations
                run_count = len(self.experimental_perturbations) if self.experimental_perturbations else 1
            
            return [(self.base_config, self.base_geometry, self.base_forcing)] * run_count
        
        # Traditional sweep parameter logic
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
        Orchestrate spinup runs based on spinup_config mode or spinup_objects.
        Returns a mapping of run_id -> profile_path for legacy mode,
        or run_id -> {'profile': path, 'config': obj, 'geometry': obj, 'forcing': obj} for new mode.
        """
        # Handle new spinup_objects approach
        if self.spinup_objects:
            return self._handle_spinup_objects(run_objects_list, client)
        
        # Handle legacy spinup_config approach
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
        spinup_forcing = self._get_spinup_forcing()
        
        # Run single spinup
        spinup_task = dask.delayed(run_spinup_simulation)(
            ("shared", spinup_config, spinup_geometry, spinup_forcing, 
             self.output_dir, self.no_progress)
        )
        
        print("Executing shared spinup...")
        spinup_result = compute(spinup_task)[0]
        
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
                spinup_forcing = self._get_spinup_forcing()  # TODO: Custom spinup mode needs rework for per-run forcing
                
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
            spinup_results = compute(spinup_tasks)[0]
        else:
            with tqdm(total=len(spinup_tasks), desc="Spinup runs", ncols=100) as pbar:
                spinup_results = compute(spinup_tasks)[0]
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
            spinup_forcing = self._get_spinup_forcing()
            
            spinup_task = dask.delayed(run_spinup_simulation)(
                (run_id, spinup_config, spinup_geometry, spinup_forcing,
                 self.output_dir, self.no_progress)
            )
            spinup_tasks.append(spinup_task)
        
        # Execute all spinup tasks
        print(f"Executing {len(spinup_tasks)} individual spinups...")
        
        if self.no_progress:
            spinup_results = compute(spinup_tasks)[0]
        else:
            with tqdm(total=len(spinup_tasks), desc="Individual spinups", ncols=100) as pbar:
                spinup_results = compute(spinup_tasks)[0]
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

    def _handle_spinup_objects(self, run_objects_list, client):
        """Handle FlowlineSpinup objects for the new 4-object architecture."""
        print("Executing FlowlineSpinup objects...")
        
        # Convert single spinup object to dict format if needed
        if isinstance(self.spinup_objects, dict):
            spinup_mapping = self.spinup_objects
        else:
            # Single spinup object: create mapping for all runs
            spinup_mapping = {}
            for i in range(len(run_objects_list)):
                run_id = self._get_run_id(i)
                spinup_mapping[run_id] = self.spinup_objects
        
        # Create Dask tasks for each unique FlowlineSpinup object
        unique_spinups = {}  # FlowlineSpinup object -> run_ids that use it
        spinup_tasks = {}    # FlowlineSpinup object -> Dask task
        
        # Group run_ids by their FlowlineSpinup object (allow sharing)
        for run_id, spinup_obj in spinup_mapping.items():
            if spinup_obj not in unique_spinups:
                unique_spinups[spinup_obj] = []
            unique_spinups[spinup_obj].append(run_id)
        
        # Create Dask tasks for each unique spinup
        for spinup_obj in unique_spinups.keys():
            # Use the first run_id for this spinup as the identifier
            first_run_id = unique_spinups[spinup_obj][0]
            task = dask.delayed(spinup_obj.generate_profile)(
                self.output_dir, first_run_id, self.no_progress
            )
            spinup_tasks[spinup_obj] = task
        
        # Execute all unique spinup tasks in parallel
        print(f"Executing {len(spinup_tasks)} unique spinup configurations...")
        task_list = list(spinup_tasks.values())
        
        if self.no_progress:
            spinup_results = compute(task_list)[0]
        else:
            with tqdm(total=len(task_list), desc="FlowlineSpinup runs", ncols=100) as pbar:
                spinup_results = compute(task_list)[0]
                for _ in spinup_results:
                    pbar.update(1)
        
        # Map results back to spinup objects
        spinup_obj_results = {}
        for i, spinup_obj in enumerate(spinup_tasks.keys()):
            result = spinup_results[i]
            profile_path = result
            if str(profile_path).startswith("ERROR"):
                raise RuntimeError(f"FlowlineSpinup failed: {profile_path}")
            
            spinup_obj_results[spinup_obj] = profile_path
        
        # Build final run_id -> profile path mapping
        run_spinup_mapping = {}
        for spinup_obj, run_ids in unique_spinups.items():
            profile_path = spinup_obj_results[spinup_obj]
            for run_id in run_ids:
                run_spinup_mapping[run_id] = profile_path
        
        print(f"FlowlineSpinup objects completed. {len(run_spinup_mapping)} runs configured.")
        return run_spinup_mapping

    def _apply_experimental_perturbations(self, run_id, config, geometry, forcing):
        """
        Apply experimental perturbations using lambda functions for relative changes.
        
        Parameters
        ----------
        run_id : str
            Run identifier
        config : FlowlineConfig
            Configuration object to perturb
        geometry : FlowlineGeometry
            Geometry object to perturb
        forcing : MassBalanceForcing
            Forcing object to perturb
            
        Returns
        -------
        tuple
            (perturbed_config, perturbed_geometry, perturbed_forcing)
        """
        perturbations = self.experimental_perturbations[run_id]
        
        # Create copies to avoid modifying originals
        perturbed_config = deepcopy(config)
        perturbed_geometry = deepcopy(geometry)
        perturbed_forcing = deepcopy(forcing)
        
        for param_path, perturbation_func in perturbations.items():
            parts = param_path.split('.')
            
            if parts[0] == 'config':
                param_name = parts[1]
                if hasattr(perturbed_config, param_name):
                    current_value = getattr(perturbed_config, param_name)
                    new_value = perturbation_func(current_value)
                    setattr(perturbed_config, param_name, new_value)
                    print(f"Applied experimental perturbation {run_id}: {param_path} {current_value} -> {new_value}")
            
            elif parts[0] == 'geometry':
                param_name = parts[1]
                if hasattr(perturbed_geometry, param_name):
                    current_value = getattr(perturbed_geometry, param_name)
                    new_value = perturbation_func(current_value)
                    setattr(perturbed_geometry, param_name, new_value)
                    print(f"Applied experimental perturbation {run_id}: {param_path} {current_value} -> {new_value}")
            
            elif parts[0] == 'forcing':
                param_name = parts[1]
                if hasattr(perturbed_forcing, param_name):
                    current_value = getattr(perturbed_forcing, param_name)
                    new_value = perturbation_func(current_value)
                    setattr(perturbed_forcing, param_name, new_value)
                    print(f"Applied experimental perturbation {run_id}: {param_path} {current_value} -> {new_value}")
        
        return perturbed_config, perturbed_geometry, perturbed_forcing

    def _create_spinup_config(self, base_config, custom_overrides=None):
        """Create spinup config by applying base spinup config and custom overrides."""
        # Check if spinup_config['config'] is a FlowlineConfig object
        base_spinup_config = self.spinup_config.get('config')
        if hasattr(base_spinup_config, '__dict__'):
            # It's a FlowlineConfig object, use it as base
            spinup_config = deepcopy(base_spinup_config)
        else:
            # Legacy dictionary format, create from base_config
            spinup_config = deepcopy(base_config)
            if base_spinup_config:
                for key, value in base_spinup_config.items():
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

    def _get_spinup_forcing(self):
        """Get spinup forcing object from spinup_config."""
        spinup_forcing = self.spinup_config.get('forcing')
        
        # Check if it's a FlowlineForcing object
        if hasattr(spinup_forcing, '__dict__') and hasattr(spinup_forcing, 'get_mass_balance'):
            # It's already a FlowlineForcing object, return a copy
            return deepcopy(spinup_forcing)
        
        # Legacy dictionary format - raise informative error
        if isinstance(spinup_forcing, dict):
            raise ValueError(
                "Spinup forcing must be a FlowlineForcing object, not a dictionary.\n"
                "Please update your spinup_config to use forcing objects:\n"
                "Example:\n"
                "  spinup_config = {\n"
                "    'mode': 'shared',\n"
                "    'enabled': True,\n"
                "    'config': FlowlineConfig(tf=500, deltout=1, ...),\n"
                "    'forcing': TemperaturePrecipitationForcing(T0=8.0, P0=2.0, tf=500, ...)\n"
                "  }"
            )
        
        # No forcing specified
        if spinup_forcing is None:
            raise ValueError(
                "Explicit spinup forcing object is required in spinup_config['forcing'].\n"
                "Example: spinup_config = {\n"
                "  'forcing': TemperaturePrecipitationForcing(T0=8.0, P0=2.0, tf=500, ...)\n"
                "}"
            )
        
        return deepcopy(spinup_forcing)

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
        
        # Determine sweep dimensions from either sweep_parameters or spinup_objects
        if self.sweep_parameters:
            sweep_dims = list(self.sweep_parameters.keys())
        elif self.spinup_objects and isinstance(self.spinup_objects, dict):
            # For spinup_objects dict, create a run dimension
            sweep_dims = ['run_id']
        else:
            sweep_dims = []
        
        def preprocess_ds(ds):
            params = json.loads(ds.attrs['run_parameters'])
            coords = {}
            for dim_key in sweep_dims:
                if dim_key == 'run_id':
                    # Special handling for run_id dimension - read from separate attribute
                    coords['run_id'] = ds.attrs.get('run_id', 'unknown')
                else:
                    # Traditional sweep parameter handling
                    keys = dim_key.split('.')
                    val = params
                    for k in keys:
                        val = val[k]
                    coord_name = dim_key.replace('.', '_')
                    coords[coord_name] = val
            
            # Store the run_parameters as a data variable to preserve per-run parameters
            if 'run_id' in coords:
                run_id = coords['run_id']
                # Create a scalar data variable for this run's parameters
                import xarray as xr
                ds = ds.assign({f'run_parameters_{run_id}': xr.DataArray(ds.attrs['run_parameters'], dims=())})
            
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
                if self.sweep_parameters:
                    # Traditional sweep parameters
                    sweep_value_lists = [self.sweep_parameters[key] for key in sweep_dims]
                    shape = [len(v) for v in sweep_value_lists]
                elif 'run_id' in sweep_dims:
                    # spinup_objects dict - simple 1D concatenation along run_id
                    shape = [len(sorted_runs)]
                else:
                    shape = []

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

                if 'run_id' in sweep_dims:
                    # Simple case: 1D concatenation along run_id
                    combined_ds = xr.open_mfdataset(
                        sorted_runs,
                        preprocess=preprocess_ds,
                        combine='by_coords'
                    )
                else:
                    # Traditional nested combine for multi-dimensional sweeps
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
            
            # If using spinup_objects, inherit ALL parameters from spinup object
            if isinstance(self.spinup_objects, dict) and run_id in self.spinup_objects:
                spinup_obj = self.spinup_objects[run_id]
                # Use ALL objects from spinup: config, geometry, forcing
                config = deepcopy(spinup_obj.config)
                geometry = deepcopy(spinup_obj.geometry)
                forcing = deepcopy(spinup_obj.forcing)
                
                # Apply spinup profile to geometry
                if run_id in run_profile_mapping:
                    geometry.profile = run_profile_mapping[run_id]
                    if hasattr(geometry, 'h_init'):
                        geometry.h_init = None
                        
            elif self.spinup_objects and not isinstance(self.spinup_objects, dict):
                # Single spinup object case - inherit everything
                config = deepcopy(self.spinup_objects.config)
                geometry = deepcopy(self.spinup_objects.geometry)
                forcing = deepcopy(self.spinup_objects.forcing)
                
                # Apply spinup profile to geometry
                if run_id in run_profile_mapping:
                    geometry.profile = run_profile_mapping[run_id]
                    if hasattr(geometry, 'h_init'):
                        geometry.h_init = None
                        
            else:
                # Legacy path: only handle profile
                if run_id in run_profile_mapping:
                    profile_path = run_profile_mapping[run_id]
                    
                    # Apply spinup profile to geometry
                    geometry = deepcopy(geometry)
                    geometry.profile = profile_path
                    if hasattr(geometry, 'h_init'):
                        geometry.h_init = None
            
            # Apply experimental perturbations if specified
            if run_id in self.experimental_perturbations:
                config, geometry, forcing = self._apply_experimental_perturbations(
                    run_id, config, geometry, forcing
                )
            
            task = dask.delayed(run_flowline_simulation)((run_id, config, geometry, forcing, 
                                                          self.output_dir, self.no_progress))
            exp_tasks.append(task)
            
        print("Executing experimental runs...")
        results = []
        if self.no_progress:
            futures = compute(exp_tasks)[0]
            for future in futures:
                results.append(future)
        else:
            with tqdm(total=len(exp_tasks), desc="Experimental runs", ncols=100) as pbar:
                futures = compute(exp_tasks)[0]
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
