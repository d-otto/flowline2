#!/usr/bin/env python3
"""
Demonstrates how to programmatically set up and run a parameter sweep using
the new FlowlineSweep class that accepts objects directly.

This example shows a mu vs gamma parameter sweep with custom post-processing.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys

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
    args = parse_sweep_cli_args("Run a programmatic flowline model parameter sweep.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Base Configuration ---
    base_config = FlowlineConfig(
        ts=0,
        tf=500,
        delx=25,
        delt=0.0125 / 16,
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
        T0=8.0,
        P0=2.0,
        gamma=6.5e-3,
        mu=0.65
    )
    
    # --- Sweep Parameters ---
    # We will sweep over the melt factor `mu` and the temperature lapse rate `gamma`.
    # For mu, we'll use a range of 0.2 centered on 0.65, with a 0.05 step.
    mu_values = np.round(np.arange(0.55, 0.75 + 0.01, 0.05), 2).tolist()
    # For gamma, a range of 0.2 is not physically plausible. We'll use a
    # range of 0.002 centered on 0.0065, with a 0.0005 step.
    gamma_values = np.round(np.arange(0.0055, 0.0075 + 0.0001, 0.0005), 4).tolist()
    
    sweep_parameters = {
        'forcing.mu': mu_values,
        'forcing.gamma': gamma_values
    }  # Total runs = 5 * 5 = 25
    
    print(f"Sweep parameters:")
    print(f"  mu values: {mu_values}")
    print(f"  gamma values: {gamma_values}")
    print(f"  Total runs: {len(mu_values) * len(gamma_values)}")
    
    # --- Run the Sweep ---
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        **get_sweep_cli_kwargs(args)
    )
    
    sweep.run()
    
    # --- Custom Post-Processing ---
    print(f"\\nSweep completed. Results saved to: {args.output_dir}")
    
    # Load and analyze results
    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"
    
    if not combined_results_path.exists():
        print("Combined results file not found. Cannot create plots.")
        return
    
    print(f"Loading combined results from: {combined_results_path}")
    ds = xr.open_dataset(combined_results_path)
    
    # The result is an xarray Dataset with dimensions for each swept parameter.
    print("\\nCombined Dataset Structure:")
    print(ds)
    
    # --- Plot 1: Length and Volume Trajectories ---
    # Calculate ice volume for each run
    # ds['w'] is 1D, so it will broadcast correctly with h(time, ..., x)
    ice_volume_m3 = (ds['h'] * ds['w'] * ds.attrs['delx']).sum(dim='x')
    ice_volume_km3 = ice_volume_m3 / 1e9
    
    # Create a facet grid for length trajectories
    g_len = (ds['edge'] / 1000).plot.line(
        x='time', col='forcing_gamma', hue='forcing_mu', col_wrap=3
    )
    g_len.fig.suptitle('Glacier Length Trajectories', y=1.03, fontsize=16)
    g_len.set_titles("γ = {value}")
    g_len.set_xlabels('Time (years)')
    g_len.set_ylabels('Length (km)')
    plt.tight_layout()
    plot_path_len = output_dir / "sweep_plot_length_trajectories.png"
    plt.savefig(plot_path_len, dpi=150)
    print(f"\\nLength trajectory plot saved to: {plot_path_len}")
    plt.close(g_len.fig)
    
    # Create a facet grid for volume trajectories
    g_vol = ice_volume_km3.plot.line(
        x='time', col='forcing_gamma', hue='forcing_mu', col_wrap=3
    )
    g_vol.fig.suptitle('Glacier Volume Trajectories', y=1.03, fontsize=16)
    g_vol.set_titles("γ = {value}")
    g_vol.set_xlabels('Time (years)')
    g_vol.set_ylabels('Volume (km³)')
    plt.tight_layout()
    plot_path_vol = output_dir / "sweep_plot_volume_trajectories.png"
    plt.savefig(plot_path_vol, dpi=150)
    print(f"Volume trajectory plot saved to: {plot_path_vol}")
    plt.close(g_vol.fig)
    
    # --- Plot 2: Sensitivity of Final State to Parameters ---
    # Extract final length and volume
    final_length_km = ds['edge'].isel(time=-1) / 1000
    final_volume_km3 = ice_volume_km3.isel(time=-1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Sensitivity of Final Glacier State to Parameters', fontsize=16)
    
    # Plot final length sensitivity
    final_length_km.plot.contourf(
        ax=axes[0],
        levels=10,
        cbar_kwargs={'label': 'Final Length (km)'}
    )
    axes[0].set_title('Final Length Sensitivity')
    axes[0].set_xlabel('Melt Factor (μ)')
    axes[0].set_ylabel('Lapse Rate (γ)')
    
    # Plot final volume sensitivity
    final_volume_km3.plot.contourf(
        ax=axes[1],
        levels=10,
        cbar_kwargs={'label': 'Final Volume (km³)'}
    )
    axes[1].set_title('Final Volume Sensitivity')
    axes[1].set_xlabel('Melt Factor (μ)')
    axes[1].set_ylabel('Lapse Rate (γ)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path_sensitivity = output_dir / "sweep_plot_sensitivity.png"
    plt.savefig(plot_path_sensitivity, dpi=150)
    print(f"Sensitivity plot saved to: {plot_path_sensitivity}")
    plt.close(fig)
    
    print("\\nCustom post-processing completed!")
    print("This example demonstrates the new unified config+run approach with:")
    print("- Direct object creation and manipulation")
    print("- Integrated CLI argument handling")
    print("- Custom post-processing and visualization")

if __name__ == "__main__":
    main()