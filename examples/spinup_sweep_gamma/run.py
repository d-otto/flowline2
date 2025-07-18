#!/usr/bin/env python3
"""
Spinup sweep example using the new unified config+run approach.

This demonstrates a parameter sweep with a spin-up stage, where each lapse rate
gets spun up to steady state before the main experimental run.
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
    args = parse_sweep_cli_args("Run a spinup sweep example with lapse rate sweep.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Base Configuration ---
    base_config = FlowlineConfig(
        ts=0,
        tf=500,  # Main run is a 500-year experiment
        delx=25,
        delt=0.00078125,  # 0.0125 / 8
        deltout=1.0,
        min_thick=1.0
    )
    
    # --- Base Geometry ---
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(
        bed_characteristic_length=10000,
        domain_extent=12000,
        x_gr_points=61,
        width=1000,
        elevation_drop=1000
    )
    
    # Create reasonable initial ice thickness profile to avoid zero thickness issues
    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr / length))
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr,
        w_geom=w_geom,
        x_init=x_gr,
        h_init=h_init  # Reasonable initial profile to avoid numerical issues
    )
    
    # --- Base Forcing ---
    base_forcing = TemperaturePrecipitationForcing(
        ts=base_config.ts,
        tf=base_config.tf,
        P0=2.0,
        T0=8.2,  # The main run uses a warmer climate
        mu=0.65,  # Set a default melt factor since it's not being swept
        gamma=0.0065  # This will be overridden by sweep parameters
    )
    
    # --- Sweep Parameters ---
    # Here, we sweep over the lapse rate `gamma`
    sweep_parameters = {
        'forcing.gamma': [0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]
    }
    
    # --- Spinup Configuration ---
    # Use shared spinup mode - all runs can share the same equilibrium state
    # since we're only varying gamma (lapse rate) in the main simulation
    
    # Create spinup config and forcing objects
    spinup_base_config = FlowlineConfig(
        ts=0,
        tf=500,  # Run for 500 years to reach steady state
        delx=25,
        delt=0.00078125,
        deltout=1,  # Always use deltout=1
        min_thick=1.0
    )
    
    spinup_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=500,
        T0=8.0,  # Use a stable climate for the spin-up
        P0=2.0,
        mu=0.65,  # Match base melt factor
        gamma=0.0065  # Use a standard lapse rate for spinup
    )
    
    spinup_config = {
        'mode': 'shared',
        'enabled': True,
        'config': spinup_base_config,
        'forcing': spinup_forcing
    }
    
    print(f"Spinup sweep setup:")
    print(f"  Lapse rate values: {sweep_parameters['forcing.gamma']}")
    print(f"  Spinup climate: T0={spinup_forcing.T0}°C, P0={spinup_forcing.P0}m/yr")
    print(f"  Main run climate: T0={base_forcing.T0}°C, P0={base_forcing.P0}m/yr")
    print(f"  Total runs: {len(sweep_parameters['forcing.gamma'])}")
    
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
    print(f"\\nSpinup sweep completed. Results saved to: {args.output_dir}")
    
    # Load and analyze results
    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"
    
    if combined_results_path.exists():
        import xarray as xr
        import matplotlib.pyplot as plt
        
        print("Creating custom analysis...")
        ds = xr.open_dataset(combined_results_path)
        
        # Plot glacier response to different lapse rates
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Glacier Response to Temperature Lapse Rate', fontsize=16)
        
        # Length trajectories
        (ds['edge'] / 1000).plot.line(x='time', hue='forcing_gamma', ax=axes[0, 0])
        axes[0, 0].set_title('Length Evolution')
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('Length (km)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume trajectories
        ice_volume_km3 = (ds['h'] * ds['w'] * ds.attrs['delx']).sum(dim='x') / 1e9
        ice_volume_km3.plot.line(x='time', hue='forcing_gamma', ax=axes[0, 1])
        axes[0, 1].set_title('Volume Evolution')
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('Volume (km³)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final state vs lapse rate
        final_length_km = ds['edge'].isel(time=-1) / 1000
        final_volume_km3 = ice_volume_km3.isel(time=-1)
        
        axes[1, 0].plot(ds['forcing_gamma'] * 1000, final_length_km, 'o-')
        axes[1, 0].set_title('Final Length vs Lapse Rate')
        axes[1, 0].set_xlabel('Lapse Rate (°C/km)')
        axes[1, 0].set_ylabel('Final Length (km)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(ds['forcing_gamma'] * 1000, final_volume_km3, 'o-', color='orange')
        axes[1, 1].set_title('Final Volume vs Lapse Rate')
        axes[1, 1].set_xlabel('Lapse Rate (°C/km)')
        axes[1, 1].set_ylabel('Final Volume (km³)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "spinup_sweep_gamma_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to: {plot_path}")
        
        # Print summary statistics
        print(f"\\nSummary:")
        print(f"  Lapse rate range: {ds['forcing_gamma'].min().values*1000:.1f} - {ds['forcing_gamma'].max().values*1000:.1f} °C/km")
        print(f"  Final length range: {final_length_km.min().values:.1f} - {final_length_km.max().values:.1f} km")
        print(f"  Final volume range: {final_volume_km3.min().values:.1f} - {final_volume_km3.max().values:.1f} km³")
    
    print("\\nThis example demonstrates:")
    print("- Spinup configuration with different climate")
    print("- Temperature lapse rate sensitivity analysis")
    print("- Custom post-processing with sensitivity plots")
    print("- Check the spinup_profiles/ directory for individual spinup results")

if __name__ == "__main__":
    main()