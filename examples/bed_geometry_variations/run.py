"""
run.py

Demonstrates how to use different bedrock geometry functions (e.g., uniform slope,
concave profile, variable width) and compares their effects on glacier evolution.

This script demonstrates:
1.  Using various bedrock geometry creation functions from `flowline.geometry`.
2.  Running multiple simulations with different bedrock shapes.
3.  Creating comparison plots to visualize the impact of bedrock geometry.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowline.flowline2d import (FlowlineConfig, TemperaturePrecipitationForcing, flowline2d)
from flowline.geometry import (FlowlineGeometry, create_uniform_slope,
                               create_concave_profile, create_variable_width)

def main():
    # --- 1. Define Output Directory ---
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Example outputs will be saved to: {output_dir}")

    # --- 2. Base Configuration (same for all runs) ---
    config = FlowlineConfig(
        ts=0, tf=300, delt=0.0125 / 16, delx=50, deltout=5.0
    )
    forcing = TemperaturePrecipitationForcing(
        T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65, ts=config.ts, tf=config.tf
    )
    base_geom_params = {
        'domain_extent': 12000,
        'x_gr_points': 61,
        'elevation_drop': 1000,
        'bed_characteristic_length': 10000,
    }

    # --- 3. Define Bedrock Geometry Scenarios ---
    scenarios = {
        'Uniform Slope': {
            'function': create_uniform_slope,
            'params': {'width': 1000, **base_geom_params}
        },
        'Concave Bed': {
            'function': create_concave_profile,
            'params': {'width': 1000, 'perturbation': -200, **base_geom_params}
        },
        'Variable Width': {
            'function': create_variable_width,
            'params': {'w_head': 1500, 'w_term': 500, **base_geom_params}
        },
    }

    # --- 4. Run Simulations for Each Bedrock Geometry ---
    results = {}
    geometries = {}
    print("Running simulations for different bedrock geometries...")
    for name, scenario in scenarios.items():
        print(f"  - Running: {name}")
        # Create bedrock geometry
        x_gr, zb_gr, w_geom = scenario['function'](**scenario['params'])
        h_init = np.maximum(0, 100 * (1 - x_gr / 5000))
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
        geometries[name] = geometry

        # Run model
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        results[name] = model.run()
    print("All simulations complete.")

    # --- 5. Create Comparison Plots ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle("Bedrock Geometry Variation Comparison", fontsize=16)

    # Plot 1: Bedrock profiles and widths
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    for name, geom in geometries.items():
        geom.setup_grid(config.delx) # Need to call this to get interpolated values
        ax1.plot(geom.x / 1000, geom.zb, label=f'{name} Bed')
        ax1b.plot(geom.x / 1000, geom.w, linestyle='--', label=f'{name} Width')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Elevation (m)')
    ax1b.set_ylabel('Width (m)')
    ax1.set_title('Initial Bedrock and Width Profiles')
    ax1.legend(loc='upper left')
    ax1b.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final glacier profiles
    ax2 = fig.add_subplot(gs[0, 1])
    for name, res in results.items():
        edge_idx = res.edge_idx[-1]
        ax2.plot(res.x / 1000, res.zb, 'k--', alpha=0.5)
        ax2.fill_between(res.x[:edge_idx] / 1000, res.zb[:edge_idx],
                         res.zb[:edge_idx] + res.h[-1, :edge_idx],
                         alpha=0.5, label=name)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title('Final Glacier Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Length evolution
    ax3 = fig.add_subplot(gs[1, :])
    for name, res in results.items():
        ax3.plot(res.t, res.edge / 1000, label=name)
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Glacier Length (km)')
    ax3.set_title('Length Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "geometry_variations_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to {plot_path}")
    # plt.show()

if __name__ == "__main__":
    main()
