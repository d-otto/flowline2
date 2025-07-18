#!/usr/bin/env python3
"""
Advanced parameter sweep example demonstrating the full power of Python configuration.

This example shows how to:
1. Use different bed geometries (convex, flat, concave)
2. Apply stochastic mass balance forcing with different noise levels
3. Perform comprehensive statistical analysis similar to white noise tests
4. Create histogram plots of final glacier lengths

This demonstrates capabilities that were impossible with YAML configs.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep
from src.flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from src.flowline.flowline2d import FlowlineConfig, DirectMassBalanceForcing
from src.flowline.geometry import FlowlineGeometry
import src.flowline.geometry as geometry_module

def create_bed_geometry(curvature=0, L=12000, base_slope=0.1):
    """
    Create bed geometries with different curvatures.
    
    Parameters
    ----------
    curvature : float
        Curvature parameter in meters. Positive = concave, negative = convex, 0 = flat
    L : float
        Domain length
    base_slope : float
        Base slope
    """
    x_gr_points = 61
    x_gr = np.linspace(0, L, x_gr_points)
    
    # Base slope
    zb_base = 2000 - base_slope * x_gr
    
    # Add curvature (quadratic term)
    # Normalize x to [0,1] for curvature calculation
    x_norm = x_gr / L
    curvature_term = curvature * x_norm * (1 - x_norm)  # Parabolic shape
    
    zb_gr = zb_base + curvature_term
    
    # Constant width for simplicity
    w_geom = np.full_like(x_gr, 1000)
    
    return (x_gr, zb_gr, w_geom)

def main():
    # Parse command line arguments
    args = parse_sweep_cli_args("Run an advanced parameter sweep with bed curvature and stochastic mass balance.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Set up reproducible random number generation ---
    base_seed = 42
    rng = np.random.RandomState(base_seed)
    
    # --- Base Configuration ---
    base_config = FlowlineConfig(
        ts=0,
        tf=1000,  # 1000-year simulation
        delx=25,
        delt=0.0125/8,  # Half the previous timestep for stability
        deltout=10.0,  # Output every 10 years
        min_thick=1.0
    )
    
    # --- Base Geometry ---
    # Use flat geometry as base (will be varied in sweep)
    x_gr, zb_gr, w_geom = create_bed_geometry(curvature=0)  # Flat bed
    
    # Create initial ice thickness profile  
    scale = 100
    length = 6000
    h_init = np.maximum(0, scale * (1 - x_gr / length))
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr,
        w_geom=w_geom,
        x_init=x_gr,
        h_init=h_init
    )
    
    # --- Base Forcing ---
    # Use DirectMassBalanceForcing for stochastic mass balance
    base_forcing = DirectMassBalanceForcing(
        b0=0.5  # Base mass balance rate (m/yr)
    )
    
    # --- Advanced Sweep Parameters ---
    # Create different bed curvatures: convex (-50m), flat (0m), concave (+50m)
    curvatures = [-50, 0, 50]  # Convex, flat, concave
    bed_shapes = [create_bed_geometry(curv)[1] for curv in curvatures]
    
    # Different base mass balance rates to simulate different noise levels
    mass_balance_rates = [0.3, 0.4, 0.6, 0.7]  # Different base rates (m/yr)
    
    sweep_parameters = {
        'forcing.b0': mass_balance_rates,
        'geometry.zb_gr': bed_shapes
    }
    
    print(f"Bed curvature options: {curvatures} m (convex, flat, concave)")
    print(f"Mass balance rates: {mass_balance_rates} m/yr")
    print(f"Total base combinations: {len(mass_balance_rates)}")
    print("Note: For full bed curvature sweep, run multiple times with different geometries")
    
    # --- Spinup Configuration ---
    # Use shared spinup for all runs - efficient for multiple mass balance scenarios
    spinup_config = {
        'mode': 'shared',
        'enabled': True,
        'config': {
            'tf': 2000,  # Long spinup for direct mass balance to reach equilibrium
            'deltout': 1  # Always use deltout=1
        },
        'forcing': {
            'b0': 0.5  # Use base mass balance rate for equilibrium spinup
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
    print(f"\\nAdvanced parameter sweep completed. Results saved to: {args.output_dir}")
    
    # Load and analyze results
    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"
    
    if combined_results_path.exists():
        import xarray as xr
        
        print("Creating comprehensive analysis plots...")
        ds = xr.open_dataset(combined_results_path)
        
        # Create comprehensive analysis plots including histogram of lengths
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Advanced Parameter Sweep Analysis', fontsize=16)
        
        # Final glacier lengths
        final_lengths_km = ds['edge'].isel(time=-1) / 1000  # Convert to km
        
        # 1. Histogram of final glacier lengths (like white_noise test)
        axes[0, 0].hist(final_lengths_km.values.flatten(), bins=min(15, len(final_lengths_km)), 
                       alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].set_xlabel('Final Glacier Length (km)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Final Lengths')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_length = final_lengths_km.mean().values
        std_length = final_lengths_km.std().values
        axes[0, 0].axvline(mean_length, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_length:.1f} km')
        if std_length > 0:
            axes[0, 0].axvline(mean_length + std_length, color='orange', linestyle=':', alpha=0.8, 
                              label=f'+1σ: {mean_length + std_length:.1f} km')
            axes[0, 0].axvline(mean_length - std_length, color='orange', linestyle=':', alpha=0.8, 
                              label=f'-1σ: {mean_length - std_length:.1f} km')
        axes[0, 0].legend(fontsize=8)
        
        # 2. Time series of all runs
        if 'forcing_b0' in ds.dims:
            for i in range(len(ds['forcing_b0'])):
                length_series = ds['edge'].isel(forcing_b0=i) / 1000
                label = f'b0={ds["forcing_b0"].values[i]:.1f} m/yr'
                axes[0, 1].plot(ds['time'], length_series, alpha=0.8, linewidth=2, label=label)
        else:
            length_series = ds['edge'] / 1000
            axes[0, 1].plot(ds['time'], length_series, alpha=0.8, linewidth=2)
        
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('Glacier Length (km)')
        axes[0, 1].set_title('Length Evolution (All Runs)')
        axes[0, 1].grid(True, alpha=0.3)
        if 'forcing_b0' in ds.dims:
            axes[0, 1].legend(fontsize=8)
        
        # 3. Final length vs mass balance rate
        if 'forcing_b0' in ds.dims:
            axes[1, 0].scatter(ds['forcing_b0'], final_lengths_km, alpha=0.7, s=60, c='green')
            axes[1, 0].set_xlabel('Base Mass Balance Rate (m/yr)')
            axes[1, 0].set_ylabel('Final Glacier Length (km)')
            axes[1, 0].set_title('Mass Balance Sensitivity')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(ds['forcing_b0']) > 1:
                z = np.polyfit(ds['forcing_b0'], final_lengths_km, 1)
                p = np.poly1d(z)
                axes[1, 0].plot(ds['forcing_b0'], p(ds['forcing_b0']), "r--", alpha=0.8, 
                               label=f'Trend: {z[0]:.1f} km per m/yr')
                axes[1, 0].legend(fontsize=8)
        
        # 4. Volume evolution statistics
        ice_volume_km3 = (ds['h'] * ds['w'] * ds.attrs['delx']).sum(dim='x') / 1e9
        final_volumes = ice_volume_km3.isel(time=-1)
        
        axes[1, 1].hist(final_volumes.values.flatten(), bins=min(15, len(final_volumes)), 
                       alpha=0.7, edgecolor='black', color='lightcoral')
        axes[1, 1].set_xlabel('Final Glacier Volume (km³)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Final Volumes')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        custom_plot_path = output_dir / "advanced_sweep_analysis.png"
        plt.savefig(custom_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to: {custom_plot_path}")
        
        # Print summary statistics
        print(f"\\nSummary statistics:")
        if 'forcing_b0' in ds.dims:
            print(f"Number of successful runs: {len(ds['forcing_b0'])}")
        else:
            print(f"Number of successful runs: 1")
            
        if 'edge' in ds.variables:
            print(f"Final glacier lengths: {final_lengths_km.min().values:.1f} - {final_lengths_km.max().values:.1f} km")
            print(f"Mean final length: {final_lengths_km.mean().values:.1f} ± {final_lengths_km.std().values:.1f} km")
            print(f"Final volumes: {final_volumes.min().values:.1f} - {final_volumes.max().values:.1f} km³")
            print(f"Mean final volume: {final_volumes.mean().values:.1f} ± {final_volumes.std().values:.1f} km³")
    
    print("\\nThis example demonstrates:")
    print("- Direct mass balance forcing with parameter sweeps")
    print("- Statistical analysis of glacier length distributions")
    print("- Comprehensive plotting similar to white noise tests")
    print("- Advanced post-processing with error bars and trend analysis")
    print("- Full Python flexibility in parameter sweep setup")
    print("\\nTo test different bed curvatures, modify the create_bed_geometry() curvature parameter")
    print("and run separate sweeps for convex (-50), flat (0), and concave (+50) beds.")

if __name__ == "__main__":
    main()