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

from flowline.sweep import FlowlineSweep

def main():
    # --- 1. Define Sweep Configuration as a Python Dictionary ---
    # This configuration is similar to `sweep_config.yml` but defined in code.
    # We will sweep over the melt factor `mu` and `elevation_drop`.
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
                    'x_gr_points': 61, 'width': 1000
                },
                'h_init_params': {'scale': 100, 'length': 5000}
            },
            'forcing': {'mode': 'TP', 'T0': 8.0, 'P0': 2.0, 'gamma': 6.5e-3}
        },
        'sweep_parameters': {
            'config.mu': [0.60, 0.65, 0.70, 0.75],          # 4 values
            'geometry.parameters.elevation_drop': [900, 1100] # 2 values
        } # Total runs = 4 * 2 = 8
    }

    # --- 2. Set Up Temporary Config File and Output Directory ---
    output_dir = Path(__file__).parent / "example_outputs" / "sweep_example_output"
    
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

    # Create a plot showing final glacier length vs. the swept parameters.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # We can use xarray's plotting capabilities to easily handle multi-dimensional data.
    # Here, we plot final length for each 'mu' value, with different lines for 'elevation_drop'.
    final_lengths = ds['edge'].isel(time=-1) / 1000 # Final length in km
    
    final_lengths.plot.line(x='config_mu', ax=ax)
    
    ax.set_title('Final Glacier Length vs. Melt Factor (mu)', fontsize=16)
    ax.set_xlabel('Melt Factor (mu)')
    ax.set_ylabel('Final Length (km)')
    ax.grid(True, alpha=0.3)
    
    plot_path = output_dir / "sweep_results_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSweep results plot saved to: {plot_path}")
    # plt.show()

if __name__ == "__main__":
    main()
