#!/usr/bin/env python3
"""
Auto steady-state demo using the new FlowlineSpinup architecture.

This demonstrates the new 4-object pattern (Config, Geometry, Forcing, Spinup)
where each parameter set generates its own steady-state profile with target 
matching, then tests response to climate perturbations.

Example scenario: 
- Sweep over melt factors (mu = 0.5, 0.6, 0.7)
- Each gets spun up with adjusted T0 to achieve similar glacier length (~8km)
- Then test response to +1°C warming for 200 years
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import json

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module

def main():
    # Parse command line arguments
    args = parse_sweep_cli_args("Auto steady-state demo with FlowlineSpinup objects.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Base Configuration for Response Testing ---
    response_config = FlowlineConfig(
        ts=0,
        tf=200,  # Response test duration: 200 years
        delx=25,
        delt=0.00078125,
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
    
    # Create reasonable initial ice thickness profile for spinup
    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr / length))
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr,
        w_geom=w_geom,
        h0=h_init
    )
    
    # --- Base Forcing for Response Testing ---
    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,
        T0=8.0,  # Will be overridden by spinup perturbations
        mu=0.6   # Will be overridden by spinup
    )
    
    # --- Create FlowlineSpinup Objects ---
    # Each parameter set gets its own FlowlineSpinup object
    melt_factors = [0.5, 0.6, 0.7, 0.8]
    spinup_objects = {}
    
    for i, mu in enumerate(melt_factors):
        run_id = f"run_{i:04d}"
        
        # Spinup configuration for this parameter set
        spinup_config = FlowlineConfig(
            ts=0,
            tf=1000,  # 1000-year spinup
            delx=25,
            delt=0.00078125,
            deltout=1.0,
            min_thick=1.0
        )
        
        # Spinup forcing with this melt factor
        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0,
            tf=1000,
            P0=2.0,
            T0=8.0,  # Will be adjusted by target matching
            mu=mu
        )
        
        # Create FlowlineSpinup with target matching (perturbations now in FlowlineSweep)
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=base_geometry,
            forcing=spinup_forcing,
            target_matching={
                'target_length': 8000,  # Target 8km glacier length
                'adjustment_parameter': 'forcing.T0',
                # Adjust T0 based on melt factor for comparable lengths
                'adjustment_function': lambda mu: 8.0 + (mu - 0.6) * 3.0,
                'tolerance': 200  # Accept ±200m from target
            }
        )
        
        spinup_objects[run_id] = spinup_obj
    
    # --- Create Experimental Perturbations ---
    # Apply +1°C warming and set response duration for all runs
    experimental_perturbations = {}
    for run_id in spinup_objects.keys():
        experimental_perturbations[run_id] = {
            'forcing.T0': lambda T0_spinup: T0_spinup + 1.0,  # +1°C warming  
            'config.tf': lambda _: 200,                       # 200-year response test
            # Note: Don't override forcing.mu - it should inherit from spinup
        }
    
    # Alternative approach: Single shared spinup for all runs
    # If you want to use the same spinup for all experiments, you can do:
    # shared_spinup = FlowlineSpinup(config=spinup_config, geometry=base_geometry, forcing=shared_forcing)
    # spinup_objects = shared_spinup  # Single object, used for all runs
    
    print(f"Auto steady-state setup:")
    print(f"  Melt factor values: {melt_factors}")
    print(f"  Target glacier length: 8000m")
    print(f"  Spinup duration: 1000 years")
    print(f"  Response test: +1°C warming for 200 years")
    print(f"  Total runs: {len(spinup_objects)}")
    print(f"  Each run: Spinup -> steady state -> perturbation -> response test")
    
    # --- Run the Sweep with FlowlineSpinup Objects ---
    sweep = FlowlineSweep(
        base_config=response_config,
        base_geometry=base_geometry,
        base_forcing=response_forcing,
        spinup_objects=spinup_objects,  # Creates runs automatically from dict keys
        experimental_perturbations=experimental_perturbations,  # Apply experimental changes
        **get_sweep_cli_kwargs(args)
    )
    
    sweep.run()
    
    # --- Custom Post-processing ---
    print(f"\\nAuto steady-state sweep completed. Results saved to: {args.output_dir}")
    
    # Load and analyze results
    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"
    
    if combined_results_path.exists():

        
        print("Creating auto steady-state analysis...")
        ds = xr.open_dataset(combined_results_path)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Auto Steady-State Response to Climate Perturbation', fontsize=16)
        
        # Length trajectories during response phase
        if 'edge' in ds.data_vars:
            (ds['edge'] / 1000).plot.line(x='time', ax=axes[0, 0])
            axes[0, 0].set_title('Length Response to +1°C Warming')
            axes[0, 0].set_xlabel('Time (years)')
            axes[0, 0].set_ylabel('Length (km)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=8.0, color='red', linestyle='--', alpha=0.7, 
                             label='Target length')
            axes[0, 0].legend()
        
        # Volume trajectories during response phase
        if 'h' in ds.data_vars and 'w' in ds.data_vars:
            ice_volume_km3 = (ds['h'] * ds['w'] * ds.attrs['delx']).sum(dim='x') / 1e9
            ice_volume_km3.plot.line(x='time', ax=axes[0, 1])
            axes[0, 1].set_title('Volume Response to +1°C Warming')
            axes[0, 1].set_xlabel('Time (years)')
            axes[0, 1].set_ylabel('Volume (km³)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Extract melt factor information from the dataset
        if 'run_id' in ds.dims:
            # Extract from preserved run parameters data variables
            melt_factors_used = []
            for run_id in ds.coords['run_id'].values:
                if 'run_parameters' in ds.data_vars:
                    val = ds['run_parameters'].sel(run_id=run_id).item()
                    params = json.loads(val)
                    melt_factors_used.append(params['forcing']['mu'])
        
        # Initial vs final states comparison
        if 'edge' in ds.data_vars:
            initial_length_km = ds['edge'].isel(time=0) / 1000
            final_length_km = ds['edge'].isel(time=-1) / 1000
            
            axes[1, 0].plot(melt_factors_used, initial_length_km, 'o-', label='Initial (after spinup)')
            axes[1, 0].plot(melt_factors_used, final_length_km, 's-', label='Final (after +1°C)')
            axes[1, 0].set_title('Length Change by Melt Factor')
            axes[1, 0].set_xlabel('Melt Factor (μ)')
            axes[1, 0].set_ylabel('Length (km)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Length change
            length_change = final_length_km - initial_length_km
            axes[1, 1].plot(melt_factors_used, length_change, 'o-', color='red')
            axes[1, 1].set_title('Length Change from +1°C Warming')
            axes[1, 1].set_xlabel('Melt Factor (μ)')
            axes[1, 1].set_ylabel('Length Change (km)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "auto_steady_state_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to: {plot_path}")
        
        # Print summary statistics
        if 'edge' in ds.data_vars:
            print(f"\\nSummary:")
            print(f"  Melt factor range: {min(melt_factors_used):.3f} - {max(melt_factors_used):.3f}")
            print(f"  Initial length range: {initial_length_km.min().values:.1f} - {initial_length_km.max().values:.1f} km")
            print(f"  Final length range: {final_length_km.min().values:.1f} - {final_length_km.max().values:.1f} km")
            print(f"  Length change range: {length_change.min().values:.1f} - {length_change.max().values:.1f} km")
    
    print("\\nThis example demonstrates:")
    print("- New 4-object architecture (Config, Geometry, Forcing, Spinup)")
    print("- Auto-generated steady-state profiles with target matching")
    print("- Parameter-specific T0 adjustment for comparable initial states")
    print("- Lambda-based experimental perturbations in FlowlineSweep")
    print("- Clean separation: FlowlineSpinup for steady-state, FlowlineSweep for experiments")
    print("- Efficient sharing of FlowlineSpinup objects")

if __name__ == "__main__":
    main()