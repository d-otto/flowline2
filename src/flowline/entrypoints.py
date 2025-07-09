from pathlib import Path
import hashlib
import copy
import json
import numpy as np

from .flowline2d import (FlowlineConfig, 
                         TemperaturePrecipitationForcing, DirectMassBalanceForcing, 
                         flowline2d)
from .geometry import FlowlineGeometry
from .io import import_from_string
from .visualization import plot_run_qc

def _deep_merge(source, destination):
    """
    Recursively merges source dict into destination dict.
    Values from source overwrite values from destination.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def _create_model_from_params(run_params):
    """Instantiates a flowline2d model from a parameter dictionary."""
    # Separate params for different components
    config_p = run_params.get('config', {})
    geometry_p = run_params.get('geometry', {})
    forcing_p = run_params.get('forcing', {}).copy()
    
    config = FlowlineConfig(**config_p)
    
    geom_func = import_from_string(geometry_p['function'])
    geom_func_params = geometry_p.get('parameters', {})
    x_gr, zb_gr, w_geom = geom_func(**geom_func_params)
    
    x_init = x_gr
    h_init_params = geometry_p.get('h_init_params')
    profile_path = geometry_p.get('profile')

    if h_init_params:
        scale = h_init_params.get('scale', 100)
        length = h_init_params.get('length', 5000)
        h_init = np.maximum(0, scale * (1 - x_gr / length))
    else:
        h_init = None
    
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_init, h_init=h_init, profile=profile_path)
    
    forcing_mode = forcing_p.pop('mode', 'TP')
    if forcing_mode == 'TP':
        forcing = TemperaturePrecipitationForcing(ts=config.ts, tf=config.tf, **forcing_p)
    elif forcing_mode == 'b':
        forcing = DirectMassBalanceForcing(**forcing_p)
    else:
        raise ValueError(f"Unknown forcing mode: {forcing_mode}")

    return flowline2d(config=config, geometry=geometry, forcing=forcing)

def run_flowline_simulation(params_tuple):
    """
    Worker function executed by Dask.
    Configures and runs a single flowline simulation.
    Can perform a spin-up run if configured.
    """
    run_idx, run_params, output_dir, spinup_config = params_tuple
    
    # Generate a unique hash for this parameter set for the filename
    params_str = json.dumps(run_params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:10]
    filename = f"run_{run_idx:04d}_{params_hash}.nc"
    output_path = Path(output_dir) / filename

    try:
        # --- Stage 1: Spin-up (if configured) ---
        if spinup_config and spinup_config.get('enabled', False):
            print(f"[{run_idx}] Spin-up enabled. Preparing spin-up run...")
            # Create spin-up parameters by overriding base params with spin-up specifics
            spinup_run_params = copy.deepcopy(run_params)
            spinup_overrides = copy.deepcopy(spinup_config)
            spinup_overrides.pop('enabled', None)  # Not a model parameter
            _deep_merge(spinup_overrides, spinup_run_params)
            print(f"[{run_idx}] Spin-up params: {json.dumps(spinup_run_params, indent=2)}")

            # Ensure spin-up itself doesn't use an input profile
            if 'profile' in spinup_run_params.get('geometry', {}):
                print(f"[{run_idx}] Removing 'profile' from spin-up geometry params.")
                del spinup_run_params['geometry']['profile']

            # Instantiate and run the spin-up model
            print(f"[{run_idx}] Instantiating spin-up model...")
            spinup_model = _create_model_from_params(spinup_run_params)
            print(f"[{run_idx}] Running spin-up model...")
            spinup_result = spinup_model.run()
            print(f"[{run_idx}] Spin-up run completed.")

            # Save spin-up profile to be used by the main run
            spinup_dir = Path(output_dir) / 'spinup_profiles'
            spinup_dir.mkdir(parents=True, exist_ok=True)
            spinup_profile_path = spinup_dir / f"spinup_{filename}"
            print(f"[{run_idx}] Saving spin-up profile to: {spinup_profile_path}")
            spinup_result.to_xarray().to_netcdf(spinup_profile_path)
            
            # The main run will now use this profile as its initial state
            if 'geometry' not in run_params:
                run_params['geometry'] = {}
            run_params['geometry']['profile'] = str(spinup_profile_path)
            if 'h_init_params' in run_params.get('geometry', {}):
                del run_params['geometry']['h_init_params']
        
        # --- Stage 2: Main Run ---
        print(f"[{run_idx}] Preparing main run...")
        print(f"[{run_idx}] Main run params: {json.dumps(run_params, indent=2)}")
        model = _create_model_from_params(run_params)
        print(f"[{run_idx}] Running main model...")
        result = model.run()
        print(f"[{run_idx}] Main run completed.")

        # Save result to xarray with comprehensive metadata
        ds = result.to_xarray()
        
        ds.attrs['run_parameters'] = json.dumps(run_params, indent=4)
        
        print(f"[{run_idx}] Saving final result to: {output_path}")
        ds.to_netcdf(output_path)

        # Generate and save QC plot
        plot_output_path = output_path.with_suffix('.png')
        print(f"[{run_idx}] Generating QC plot: {plot_output_path}")
        plot_run_qc(ds, plot_output_path)
        
        print(f"[{run_idx}] Run successful.")
        return str(output_path)

    except Exception as e:
        error_file = output_path.with_suffix('.error')
        with open(error_file, 'w') as f:
            import traceback
            f.write(f"Error running simulation {run_idx}:\n")
            f.write(f"Parameters:\n{json.dumps(run_params, indent=4)}\n\n")
            f.write(traceback.format_exc())
        return f"ERROR: See {error_file.name}"
