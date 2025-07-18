#!/usr/bin/env python3
"""
Parameter sweep example using the new unified config+run approach.

This demonstrates how to set up a parameter sweep by creating FlowlineConfig,
FlowlineGeometry, and MassBalanceForcing objects directly in the script,
then passing them to FlowlineSweep.

This approach enables:
- Complex parameter generation (e.g., using numpy RNG, distributions)
- Direct object manipulation
- Custom post-processing per experiment
- Full Python power in configuration
"""

from pathlib import Path
import sys
import numpy as np

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep
from src.flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from src.flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from src.flowline.geometry import FlowlineGeometry
import src.flowline.geometry as geometry_module

def main():
    # Parse command line arguments
    args = parse_sweep_cli_args("Run a parameter sweep example with temperature sweep.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Base Configuration ---
    base_config = FlowlineConfig(
        ts=0,
        tf=100,  # Main run is a 100-year experiment
        delx=25,
        delt=0.0015625,  # 0.0125/8
        deltout=1.0,
        min_thick=1.0
    )
    
    # --- Base Geometry ---
    # Create geometry using the geometry module function
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(
        bed_characteristic_length=10000,
        domain_extent=12000,
        x_gr_points=61,
        width=1000,
        elevation_drop=1000
    )
    
    # Create initial ice thickness profile
    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr / length))
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr,
        w_geom=w_geom,
        x_init=x_gr,
        h_init=h_init
    )
    
    # --- Base Forcing ---
    base_forcing = TemperaturePrecipitationForcing(
        ts=base_config.ts,
        tf=base_config.tf,
        P0=2.0,
        T0=8.5  # Base temperature
    )
    
    # --- Sweep Parameters ---
    # This is where the power of Python configuration shines
    # You can generate complex parameter sets programmatically
    sweep_parameters = {
        'forcing.T0': [7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0]
    }
    
    # --- Spinup Configuration ---
    # Use shared spinup - all temperature scenarios start from same equilibrium
    spinup_config = {
        'mode': 'shared',
        'enabled': True,
        'config': {
            'tf': 500,  # Run for 500 years to reach steady state
            'deltout': 1  # Always use deltout=1
        },
        'forcing': {
            'T0': 7.0,  # Use stable climate for spinup
            'P0': 2.0   # Match base precipitation
        }
    }
    
    # --- Run the Sweep ---
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        spinup_config=spinup_config,
        **get_sweep_cli_kwargs(args)
    )
    
    sweep.run()
    
    # --- Custom Post-processing ---
    # This is where you can add experiment-specific analysis
    print(f"\\nParameter sweep completed. Results saved to: {args.output_dir}")
    print("Check the output directory for:")
    print("- combined_results.nc: Combined results from all runs")
    print("- Individual NetCDF files: Results from each parameter combination")
    print("- QC plots: Visual validation of results")
    print("- config.json: Configuration and environment info")
    
    # Example of how you might add custom analysis:
    # combined_results_path = Path(args.output_dir) / "combined_results.nc"
    # if combined_results_path.exists():
    #     import xarray as xr
    #     ds = xr.open_dataset(combined_results_path)
    #     # Add your custom analysis here
    #     print(f"Final glacier lengths: {ds['edge'].isel(time=-1).values / 1000} km")

if __name__ == "__main__":
    main()