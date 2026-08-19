from pathlib import Path
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
    """Instantiates a flowline2d model from a parameter dictionary.
    
    NOTE: Currently unused but kept for potential future dictionary-based interfaces.
    """
    # Separate params for different components
    config_p = run_params.get('config', {})
    geometry_p = run_params.get('geometry', {})
    forcing_p = run_params.get('forcing', {}).copy()
    
    config = FlowlineConfig(**config_p)
    
    geom_func_str = geometry_p.get('function')
    if not geom_func_str:
        raise ValueError("Missing 'geometry.function' in parameters.")
    geom_func = import_from_string(geom_func_str)
    geom_func_params = geometry_p.get('parameters', {})
    x_gr, zb_gr, w_geom = geom_func(**geom_func_params)
    
    h_init_params = geometry_p.get('h_init_params')
    profile_path = geometry_p.get('profile')

    if profile_path is not None:
        geometry = FlowlineGeometry.from_profile(profile_path, x_gr, zb_gr, w_geom)
    elif h_init_params:
        scale = h_init_params.get('scale', 100)
        length = h_init_params.get('length', 5000)
        h_init = np.maximum(0, scale * (1 - x_gr / length))
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=h_init)
    else:
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=np.zeros_like(x_gr))
    
    forcing_mode = forcing_p.pop('mode', 'TP')
    if forcing_mode == 'TP':
        forcing = TemperaturePrecipitationForcing(ts=config.ts, tf=config.tf, **forcing_p)
    elif forcing_mode == 'b':
        forcing = DirectMassBalanceForcing(**forcing_p)
    else:
        raise ValueError(f"Unknown forcing mode: {forcing_mode}")

    return flowline2d(config=config, geometry=geometry, forcing=forcing)

def run_spinup_simulation(params_tuple):
    """
    Dedicated spinup worker function executed by Dask.
    Runs a single spinup simulation and returns the profile path.
    """
    run_id, config, geometry, forcing, output_dir, no_progress = params_tuple
    
    # Generate filename using run_id
    params_dict = {
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'geometry': {'type': type(geometry).__name__, 'x_gr_shape': geometry.x_gr.shape if hasattr(geometry, 'x_gr') else None},
        'forcing': forcing.__dict__ if hasattr(forcing, '__dict__') else str(forcing)
    }
    filename = f"spinup_{run_id}.nc"
    
    spinup_dir = Path(output_dir) / 'spinup_profiles'
    spinup_dir.mkdir(parents=True, exist_ok=True)
    spinup_profile_path = spinup_dir / filename
    
    try:
        print(f"[{run_id}] Running spinup simulation...")
        
        spinup_geometry = copy.deepcopy(geometry)

        # Instantiate and run the spinup model
        print(f"[{run_id}] Instantiating spinup model...")
        spinup_model = flowline2d(config=config, geometry=spinup_geometry, forcing=forcing)
        print(f"[{run_id}] Running spinup model...")
        spinup_result = spinup_model.run(no_progress=no_progress)
        print(f"[{run_id}] Spinup run completed.")
        
        # Check if spinup produced ice formation
        if not spinup_result.no_error:
            raise Exception(f"Spinup run failed with errors")
        
        # Check for ice formation (using min_thick threshold)
        max_thickness = np.nanmax(spinup_result.h[-1, :])
        min_thick = getattr(config, 'min_thick', 0.1)  # Default 0.1m threshold
        if max_thickness < min_thick:
            raise Exception(f"Spinup run failed: no ice formation (max thickness: {max_thickness:.3f}m < {min_thick}m threshold)")
        
        print(f"[{run_id}] Spinup validation successful (max thickness: {max_thickness:.1f}m)")
        
        # Process and save spinup results
        spinup_ds = spinup_result.to_xarray()
        spinup_ds.attrs['run_parameters'] = json.dumps(params_dict, indent=4, default=str)
        spinup_ds.attrs['spinup_run_id'] = str(run_id)
        
        # Generate and save spinup QC plot
        spinup_plot_output_path = spinup_profile_path.with_suffix('.png')
        print(f"[{run_id}] Generating spinup QC plot: {spinup_plot_output_path}")
        plot_run_qc(spinup_ds, spinup_plot_output_path)
        
        # Save spinup profile
        print(f"[{run_id}] Saving spinup profile to: {spinup_profile_path}")
        spinup_ds.to_netcdf(spinup_profile_path)
        
        print(f"[{run_id}] Spinup successful.")
        return str(spinup_profile_path)
        
    except Exception as e:
        error_file = spinup_profile_path.with_suffix('.error')
        with open(error_file, 'w') as f:
            import traceback
            f.write(f"Error running spinup simulation {run_id}:\n")
            f.write(f"Parameters:\n{json.dumps(params_dict, indent=4, default=str)}\n\n")
            f.write(traceback.format_exc())
        return f"ERROR: See {error_file.name}"

def run_flowline_simulation(params_tuple):
    """
    Worker function executed by Dask that works with actual objects.
    Configures and runs a single experimental flowline simulation using FlowlineConfig,
    FlowlineGeometry, and MassBalanceForcing objects.
    
    NOTE: This function no longer handles spinup - use run_spinup_simulation for that.
    """
    run_id, config, geometry, forcing, output_dir, no_progress = params_tuple
    
    # Generate filename using run_id
    params_dict = {
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'geometry': {'type': type(geometry).__name__, 'x_gr_shape': geometry.x_gr.shape if hasattr(geometry, 'x_gr') else None},
        'forcing': forcing.__dict__ if hasattr(forcing, '__dict__') else str(forcing)
    }
    filename = f"run_{run_id}.nc"
    output_path = Path(output_dir) / filename

    try:
        print(f"[{run_id}] Preparing experimental run...")
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        print(f"[{run_id}] Running experimental model...")
        result = model.run(no_progress=no_progress)
        print(f"[{run_id}] Experimental run completed.")

        # Process and save experimental run results
        ds = result.to_xarray()
        ds.attrs['run_parameters'] = json.dumps(params_dict, indent=4, default=str)
        ds.attrs['run_id'] = str(run_id)

        # Generate and save QC plot
        plot_output_path = output_path.with_suffix('.png')
        print(f"[{run_id}] Generating QC plot: {plot_output_path}")
        plot_run_qc(ds, plot_output_path)
        
        # Save final result to NetCDF
        print(f"[{run_id}] Saving experimental result to: {output_path}")
        ds.to_netcdf(output_path)
        
        print(f"[{run_id}] Experimental run successful.")
        return str(output_path)

    except Exception as e:
        error_file = output_path.with_suffix('.error')
        with open(error_file, 'w') as f:
            import traceback
            f.write(f"Error running experimental simulation {run_id}:\n")
            f.write(f"Parameters:\n{json.dumps(params_dict, indent=4, default=str)}\n\n")
            f.write(traceback.format_exc())
        return f"ERROR: See {error_file.name}"

