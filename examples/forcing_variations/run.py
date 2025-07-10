"""
run.py

Demonstrates how to configure and use different mass balance forcing scenarios.

This script demonstrates:
1.  Performing a spin-up run to get a steady-state initial condition.
2.  Using `TemperaturePrecipitationForcing` with a warming trend.
3.  Using `DirectMassBalanceForcing` with a step change and with white noise.
4.  Comparing glacier responses to different forcing types.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from flowline.flowline2d import (FlowlineConfig, TemperaturePrecipitationForcing,
                               DirectMassBalanceForcing, flowline2d)
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline.diagnostics import calc_ela

def run_spinup(config, geometry, forcing):
    """Helper function to run the model to a steady state."""
    print("Running spin-up to find steady state...")
    model_ss = flowline2d(config=config, geometry=geometry, forcing=forcing)
    result_ss = model_ss.run()
    print("Spin-up complete.")
    return result_ss

def main():
    # --- 1. Define Output Directory ---
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Example outputs will be saved to: {output_dir}")

    # --- 2. Spin-up to Steady State ---
    # We first run the model for a long time to get a stable glacier.
    # This provides a consistent starting point for our experiments.
    spinup_config = FlowlineConfig(
        ts=0, tf=1000, delt=0.0125 / 16, delx=50, deltout=10
    )
    geom_params = {
        'domain_extent': 12000, 'x_gr_points': 61, 'elevation_drop': 1000,
        'width': 1000, 'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = create_uniform_slope(**geom_params)
    h_init = np.maximum(0, 100 * (1 - x_gr / 5000))
    spinup_geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
    spinup_forcing = TemperaturePrecipitationForcing(
        T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65, ts=spinup_config.ts, tf=spinup_config.tf
    )

    # Use the spin-up result as the initial condition for all experiments.
    # The profile object itself can be passed to FlowlineGeometry.
    steady_state_profile = run_spinup(spinup_config, spinup_geometry, spinup_forcing)
    
    # --- 3. Define Forcing Scenarios for a 200-year experiment ---
    exp_config = FlowlineConfig(
        ts=0, tf=200, delt=0.0125 / 16, delx=50, deltout=2.0
    )
    nyears = int(np.ceil(exp_config.tf - exp_config.ts))
    
    # The steady-state mass balance profile from the end of the spin-up.
    # We use this as the base for `DirectMassBalanceForcing`.
    ss_b_profile = steady_state_profile.b_profile[-1, :]

    # Scenario definitions
    scenarios = {
        'Control (T-P)': {
            'forcing_class': TemperaturePrecipitationForcing,
            'params': {'T0': 8.0, 'P0': 2.0, 'gamma': 6.5e-3, 'mu': 0.65},
            'forcing_ts': None
        },
        'Warming Trend (+2°C/100yr)': {
            'forcing_class': TemperaturePrecipitationForcing,
            'params': {
                'T0': 8.0, 'P0': 2.0, 'gamma': 6.5e-3, 'mu': 0.65,
                'temp': np.concatenate([np.linspace(0, 2.0, 100), np.full(nyears - 100, 2.0)])
            },
            'forcing_ts': np.concatenate([np.linspace(0, 2.0, 100), np.full(nyears - 100, 2.0)])
        },
        'Step Change (-0.5 m/yr)': {
            'forcing_class': DirectMassBalanceForcing,
            'params': {
                'b0': ss_b_profile,
                'bp': np.concatenate([np.zeros(50), np.full(nyears-50, -0.5)])
            },
            'forcing_ts': np.concatenate([np.zeros(50), np.full(nyears-50, -0.5)])
        },
        'White Noise (std=0.5)': {
            'forcing_class': DirectMassBalanceForcing,
            'params': {
                'b0': ss_b_profile,
                'bp': np.random.normal(0, 0.5, nyears)
            },
            'forcing_ts': np.random.normal(0, 0.5, nyears) # Recreate for plot
        }
    }

    # --- 4. Run Simulations for Each Scenario ---
    results = {}
    print("\nRunning simulations for different forcing scenarios...")
    for name, scenario in scenarios.items():
        print(f"  - Running: {name}")
        # Initialize geometry from the steady-state profile.
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, profile=steady_state_profile)
        
        # Initialize forcing
        forcing_params = scenario['params']
        forcing_params.update({'ts': exp_config.ts, 'tf': exp_config.tf})
        forcing = scenario['forcing_class'](**forcing_params)

        model = flowline2d(config=exp_config, geometry=geometry, forcing=forcing)
        results[name] = model.run()
    print("All simulations complete.")

    # --- 5. Diagnostic ELA Calculation ---
    # Use the diagnostic tools to calculate the ELA for the control scenario.
    # Note: P0 and gamma must be converted to units expected by calc_ela (mm and °C/km).
    control_params = scenarios['Control (T-P)']['params']
    diag_ela = calc_ela(
        P0=control_params['P0'] * 1000, # m -> mm
        T0=control_params['T0'],
        gamma=control_params['gamma'] * 1000, # C/m -> C/km
        mu=control_params['mu']
    )
    print(f"\nDiagnostic ELA Calculation (for Control scenario): {diag_ela:.2f} m")
    
    # --- 6. Create Comparison Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Mass Balance Forcing Comparison", fontsize=16)

    # Plot 1: Forcing time series
    ax = axes[0]
    time_axis = np.arange(exp_config.ts, exp_config.tf, 1)
    for name, scenario in scenarios.items():
        if scenario['forcing_ts'] is not None:
            # Pad the forcing time series to match the full time axis if needed
            ts = scenario['forcing_ts']
            if len(ts) < len(time_axis):
                ts = np.pad(ts, (0, len(time_axis) - len(ts)), 'edge')
            ax.plot(time_axis, ts, label=name)
    ax.set_ylabel('Forcing Anomaly (m/yr or °C)')
    ax.set_title('Applied Forcing Perturbations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Length evolution
    ax = axes[1]
    for name, res in results.items():
        ax.plot(res.t, res.edge / 1000, label=name)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Glacier Length (km)')
    ax.set_title('Glacier Length Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "forcing_variations_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to {plot_path}")
    # plt.show()

if __name__ == "__main__":
    main()
