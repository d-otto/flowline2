"""
example_sweep.py

Demonstrates how to programmatically set up and run a parameter sweep using
the `FlowlineSweep` class.

This script demonstrates:
1.  Creating a sweep configuration as a Python dictionary.
2.  Writing the configuration to a temporary YAML file.
3.  Instantiating and running `FlowlineSweep`.
4.  Loading the combined results from the output NetCDF file.
5.  Creating a plot from the aggregated sweep results.
"""
from pathlib import Path
import yaml
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
import argparse

from flowline.sweep import FlowlineSweep

def main():
    parser = argparse.ArgumentParser(
        description="Run a programmatic flowline model parameter sweep."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).parent / "example_outputs" / "sweep_example_output"),
        help="Directory to save sweep results."
    )
    args = parser.parse_args()

    # --- 1. Define Sweep Configuration as a Python Dictionary ---
    # This configuration is similar to `sweep_config.yml` but defined in code.
    # We will sweep over the melt factor `mu` and the temperature lapse rate `gamma`.
    # For mu, we'll use a range of 0.2 centered on 0.65, with a 0.05 step.
    mu_values = np.round(np.arange(0.55, 0.75 + 0.01, 0.05), 2).tolist()
    # For gamma, a range of 0.2 is not physically plausible. We'll use a
    # range of 0.002 centered on 0.0065, with a 0.0005 step.
    gamma_values = np.round(np.arange(0.0055, 0.0075 + 0.0001, 0.0005), 4).tolist()

    sweep_config = {
        'base_parameters': {
            'config': {
                'ts': 0, 'tf': 100, 'delx': 100, 'delt': 0.0125 / 4,
                'deltout': 10.0, 'min_thick': 1.0
            },
            'geometry': {
                'function': 'flowline.geometry.create_uniform_slope',
                'parameters': {
                    'bed_characteristic_length': 10000, 'domain_extent': 12000,
                    'x_gr_points': 61, 'width': 1000, 'elevation_drop': 1000
                },
                'h_init_params': {'scale': 100, 'length': 5000}
            },
            'forcing': {'mode': 'TP', 'T0': 8.0, 'P0': 2.0, 'gamma': 6.5e-3, 'mu': 0.65}
        },
        'sweep_parameters': {
            'forcing.mu': mu_values,
            'forcing.gamma': gamma_values
        } # Total runs = 5 * 5 = 25
    }

    # --- 2. Set Up Temporary Config File and Output Directory ---
    output_dir = Path(args.output_dir)
    
    # Using a temporary file for the config is a clean way to pass it to FlowlineSweep
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml', dir='.') as tmp:
        yaml.dump(sweep_config, tmp)
        config_filepath = tmp.name
    
    print(f"Sweep config written to temporary file: {config_filepath}")
    print(f"Sweep output will be saved to: {output_dir}")

    # --- 3. Instantiate and Run the Sweep ---
    # We can specify the number of workers (cores) to use.
    sweep = FlowlineSweep(
        config_file=config_filepath,
        output_dir=output_dir,
        workers=4  # Use 4 cores, or set to None to use all available
    )
    
    # This will execute all 8 simulations in parallel.
    sweep.run()
    
    # Clean up the temporary config file
    os.remove(config_filepath)
    print(f"Temporary config file {config_filepath} removed.")

    # --- 4. Load and Plot Combined Results ---
    combined_results_path = output_dir / "combined_results.nc"
    if not combined_results_path.exists():
        print("Combined results file not found. Cannot create plot.")
        return

    print(f"Loading combined results from: {combined_results_path}")
    ds = xr.open_dataset(combined_results_path)
    
    # The result is an xarray Dataset with dimensions for each swept parameter.
    print("\nCombined Dataset Structure:")
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
    g_len.set_titles("γ = {value:.4f}")
    g_len.set_xlabels('Time (years)')
    g_len.set_ylabels('Length (km)')
    plt.tight_layout()
    plot_path_len = output_dir / "sweep_plot_length_trajectories.png"
    plt.savefig(plot_path_len, dpi=150)
    print(f"\nLength trajectory plot saved to: {plot_path_len}")
    plt.close(g_len.fig)
    
    # Create a facet grid for volume trajectories
    g_vol = ice_volume_km3.plot.line(
        x='time', col='forcing_gamma', hue='forcing_mu', col_wrap=3
    )
    g_vol.fig.suptitle('Glacier Volume Trajectories', y=1.03, fontsize=16)
    g_vol.set_titles("γ = {value:.4f}")
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

if __name__ == "__main__":
    main()
