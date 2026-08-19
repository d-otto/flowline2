"""
run.py

Demonstrates different ways to initialize the ice thickness profile (the
glacier geometry) for a simulation.

This script demonstrates:
1.  Initializing from a simple functional form (a wedge shape).
2.  Initializing from the result of a previous "spin-up" simulation.
3.  Initializing from a pickled simulation result file saved to disk.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowline.flowline2d import (FlowlineConfig, TemperaturePrecipitationForcing, flowline2d)
from flowline.geometry import FlowlineGeometry, create_uniform_slope

def run_spinup(config, geometry, forcing):
    """Helper function to run a spin-up simulation."""
    print("Running spin-up to generate an initial glacier profile...")
    model = flowline2d(config=config, geometry=geometry, forcing=forcing)
    result = model.run()
    print("Spin-up complete.")
    return result

def main():
    # --- 1. Define Output Directory and Common Parameters ---
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Example outputs will be saved to: {output_dir}")

    # Common bedrock geometry for all scenarios
    bed_geom_params = {
        'domain_extent': 12000, 'x_gr_points': 61, 'elevation_drop': 1000,
        'width': 1000, 'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = create_uniform_slope(**bed_geom_params)

    # Common forcing for all scenarios
    forcing = TemperaturePrecipitationForcing(
        T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65, ts=0, tf=100
    )

    # Common model configuration for the main experimental runs
    exp_config = FlowlineConfig(
        ts=0, tf=100, delt=0.0125 / 16, delx=50, deltout=5.0
    )

    # --- 2. Generate a Spin-up Profile for Scenarios 2 & 3 ---
    spinup_config = FlowlineConfig(
        ts=0, tf=500, delt=0.0125 / 16, delx=50, deltout=10
    )
    spinup_forcing = TemperaturePrecipitationForcing(
        T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65, ts=spinup_config.ts, tf=spinup_config.tf
    )
    # Start spin-up from a simple wedge
    h_init_wedge = np.maximum(0, 100 * (1 - x_gr / 5000))
    spinup_geom_initial = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=h_init_wedge)
    
    # Run spin-up to get a more realistic profile
    spinup_result = run_spinup(spinup_config, spinup_geom_initial, spinup_forcing)
    
    # Save the spin-up result to a file for Scenario 3
    profile_path = output_dir / "spinup_profile.nc"
    spinup_result.to_xarray().to_netcdf(profile_path)
    print(f"Spin-up profile saved to: {profile_path}")

    # --- 3. Define Glacier Geometry Initialization Scenarios ---
    scenarios = {}
    
    # Scenario 1: Initialized from a simple wedge function
    h_wedge = np.maximum(0, 200 * (1 - x_gr / 8000)) # A different wedge
    scenarios['From Function (Wedge)'] = FlowlineGeometry(
        x_gr, zb_gr, w_geom, h0=h_wedge
    )

    # Scenario 2: Initialized from the saved spin-up profile (same file as Scenario 3)
    scenarios['From Spin-up Profile'] = FlowlineGeometry.from_profile(
        profile_path, x_gr, zb_gr, w_geom
    )

    # Scenario 3: Initialized from the saved spin-up profile file
    scenarios['From Profile File'] = FlowlineGeometry.from_profile(
        profile_path, x_gr, zb_gr, w_geom
    )
    
    # --- 4. Run Simulations for Each Initialization ---
    results = {}
    initial_profiles = {}
    print("\nRunning simulations for different initial glacier geometries...")
    for name, geometry in scenarios.items():
        print(f"  - Running: {name}")
        model = flowline2d(config=exp_config, geometry=geometry, forcing=forcing)
        # Store initial h profile for plotting
        initial_profiles[name] = model.h0.copy()
        results[name] = model.run()
    print("All simulations complete.")

    # --- 5. Create Comparison Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Glacier Geometry Initialization Comparison", fontsize=16)

    # Get the computational grid from one of the results (they are all the same)
    model_x = results['From Function (Wedge)'].x
    model_zb = results['From Function (Wedge)'].zb

    # Plot 1: Initial ice thickness profiles
    ax = axes[0]
    ax.plot(model_x / 1000, model_zb, 'k-', linewidth=2, label='Bedrock')
    for name, h0 in initial_profiles.items():
        ax.plot(model_x / 1000, model_zb + h0, label=name, linestyle='--')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Initial Ice Thickness Profiles')
    ax.set_xlabel('Distance (km)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Length evolution from each start
    ax = axes[1]
    for name, res in results.items():
        ax.plot(res.t, res.edge / 1000, label=name)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Glacier Length (km)')
    ax.set_title('Length Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "glacier_geometry_variations_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison plot saved to {plot_path}")
    # plt.show()


if __name__ == "__main__":
    main()
