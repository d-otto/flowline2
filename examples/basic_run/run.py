"""
run.py

A simple, end-to-end example of setting up and running a single flowline
simulation.

This script demonstrates:
1.  Importing necessary components from the `flowline` package.
2.  Defining model configuration (`FlowlineConfig`).
3.  Creating a glacier geometry (`FlowlineGeometry`, `create_uniform_slope`).
4.  Setting up mass balance forcing (`TemperaturePrecipitationForcing`).
5.  Instantiating and running the `flowline2d` model.
6.  Plotting and saving the results.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# --- Import Flowline Components ---
from flowline.flowline2d import (FlowlineConfig, TemperaturePrecipitationForcing, flowline2d)
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline import diagnostics as diag
from flowline.visualization import plot_run_qc

def main():
    # --- 1. Define Output Directory ---
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Example outputs will be saved to: {output_dir}")

    # --- 2. Configure the Model ---
    # These parameters are based on values from the test suite for a stable run.
    config = FlowlineConfig(
        ts=0,
        tf=500,           # Run for 500 years
        delt=0.0125 / 16, # Time step (years)
        delx=50,          # Grid spacing (meters)
        deltout=5.0,      # Output frequency (years)
    )

    # --- 3. Create Glacier Geometry ---
    # We'll use a simple, uniform slope for the glacier bed.
    geom_params = {
        'domain_extent': 12000,
        'x_gr_points': 61,
        'elevation_drop': 1000,
        'width': 1000,
        'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = create_uniform_slope(**geom_params)

    # Define an initial ice thickness profile (a simple wedge shape).
    h_init = np.maximum(0, 100 * (1 - x_gr / 5000))

    # Instantiate the geometry object.
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)

    # --- 4. Set Up Mass Balance Forcing ---
    # We'll use temperature-precipitation forcing.
    forcing = TemperaturePrecipitationForcing(
        T0=8.0,
        P0=2.0,
        gamma=6.5e-3, # Lapse rate
        mu=0.65,      # Melt factor
        ts=config.ts,
        tf=config.tf,
    )

    # --- 5. Instantiate and Run the Model ---
    print("Initializing and running the flowline model...")
    model = flowline2d(config=config, geometry=geometry, forcing=forcing)
    result = model.run()
    print("Model run complete.")

    # --- 6. Calculate and Print Diagnostic Statistics ---
    # The diagnostics module provides tools for analyzing model output.
    diag_stats = diag.calc_diag(result)
    print("\n--- Diagnostic Statistics ---")
    print(diag_stats)
    print("---------------------------\n")

    # --- 7. Save and Plot Results ---
    # Save the full results to a NetCDF file.
    result_path = output_dir / "basic_run_result.nc"
    ds = result.to_xarray()
    ds.to_netcdf(result_path)
    print(f"Results saved to {result_path}")

    # Create a standard QC plot for the run.
    plot_path = output_dir / "basic_run_qc.png"
    plot_run_qc(ds, plot_path)
    print(f"QC plot saved to {plot_path}")

if __name__ == "__main__":
    main()
