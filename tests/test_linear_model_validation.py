# -*- coding: utf-8 -*-
"""
Test suite for validating the flowline model against the linear model.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import tempfile
import xarray as xr

# Import the module under test
from flowline.flowline2d import FlowlineConfig
from flowline.forcing import TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline.linear_model import LinearModel, calc_Leq
from flowline.spinup import FlowlineSpinup, calculate_response_time
from flowline.sweep import FlowlineSweep

# Create output directory for QC figures
QC_FIGURE_DIR = Path("tests/qc_figures")
QC_FIGURE_DIR.mkdir(exist_ok=True)


class TestFlowlineLinearModelValidation:
    """Test validation of flowline model against linear model predictions"""

    def test_flowline_vs_linear_model_warming_response(self):
        """
        Test that flowline model equilibrium response to warming matches linear model.
        
        1. Use FlowlineSweep to manage a spinup and a subsequent warming experiment.
        2. Extract parameters for the linear model from the steady-state portion of the run.
        3. Compare the final length from the flowline model's warming experiment with the
           prediction from the linear model.
        """
        output_dir = QC_FIGURE_DIR / 'linear_model_validation_output'
        output_dir.mkdir(exist_ok=True)

        # ========== 1. Configure and Run the Sweep ==========
        
        simulation_time = 1000
        base_config = FlowlineConfig(ts=0, tf=simulation_time, delx=25, delt=0.0125/16, deltout=1)
        
        domain_extent = 15000
        x_gr, zb_gr, w_geom = create_uniform_slope(
            domain_extent=domain_extent, x_gr_points=int(domain_extent/base_config.delx) + 1,
            elevation_drop=1000, width=1000, bed_characteristic_length=domain_extent
        )
        h_init = np.maximum(0, 150 * (1 - x_gr / 8000))
        base_geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=h_init)
        
        base_forcing = TemperaturePrecipitationForcing(
            T0=8., P0=2.0, gamma=6.5e-3, mu=0.6, ts=base_config.ts, tf=base_config.tf
        )

        spinup_obj = FlowlineSpinup(config=base_config, geometry=base_geometry, forcing=base_forcing)

        warming = 0.5
        experimental_perturbations = {
            'run_0000': {'forcing.T0': lambda T0: T0 + warming}
        }

        sweep = FlowlineSweep(
            base_config=base_config,
            base_geometry=base_geometry,
            base_forcing=base_forcing,
            sweep_parameters={},
            spinup_objects={'run_0000': spinup_obj},
            experimental_perturbations=experimental_perturbations,
            output_dir=str(output_dir),
            workers=1,
            no_progress=True
        )
        sweep.run()

        # ========== 2. Load Results and Extract Parameters ==========
        
        # Load experimental results
        results_file = output_dir / "combined_results.nc"
        assert results_file.exists(), "Sweep did not produce a results file."
        ds = xr.open_dataset(results_file).squeeze('run_id', drop=True)
        
        # Load spinup results to extract steady-state parameters
        spinup_file = output_dir / "spinup_profiles" / "spinup_spinup_run_0000.nc"
        assert spinup_file.exists(), "Spinup file not found."
        spinup_ds = xr.open_dataset(spinup_file)
        
        # Extract parameters from final spinup state (steady-state)
        initial_length = spinup_ds['edge'].values[-1]
        initial_edge_idx = int(spinup_ds['edge_idx'].values[-1])

        ela_idx = np.argmin(np.abs(spinup_ds['x'].values - spinup_ds['ela'].values[-1]))
        H = np.mean(spinup_ds['h'].values[-1, ela_idx:initial_edge_idx])
        
        # Get mass balance at terminus for diagnostics
        bt = spinup_ds['b_profile'].values[-1, max(0, initial_edge_idx - 1)]
        
        tau = calculate_response_time(
            h=spinup_ds['h'].values[-1, :],
            b=spinup_ds['b_profile'].values[-1, :],
            delx=spinup_ds.attrs['delx'],
            edge_idx=initial_edge_idx,
            ela_idx=ela_idx
        )

        # Diagnostic output
        print(f"\n=== Flowline Model Parameters ===")
        print(f"H (mean thickness ELA-terminus): {H:.2f} m")
        print(f"bt (terminus mass balance): {bt:.4f} m/yr")
        print(f"tau (response time): {tau:.1f} years")
        print(f"L_bar (initial length): {initial_length:.1f} m")

        linear_model = LinearModel(L_bar=initial_length, H=H, tau=tau, dt=1.0)

        # ========== 3. Calculate Linear Prediction and Compare ==========
        
        mass_balance_change = -base_forcing.mu * warming
        
        # Calculate equilibrium length change using calc_Leq function
        # Calculate glacier area from initial conditions
        x_vals = spinup_ds['x'].values
        w_vals = spinup_ds['w'].values if 'w' in spinup_ds else np.mean(spinup_ds['w'].values) if 'w' in spinup_ds.data_vars else 1000  # fallback to uniform width
        if np.ndim(w_vals) == 0:
            w_vals = np.full_like(x_vals, w_vals)
        
        # Calculate area up to terminus
        delx = spinup_ds.attrs['delx']
        glacier_area = np.sum(w_vals[:initial_edge_idx]) * delx
        
        leq_change = calc_Leq(
            A=glacier_area,
            w=w_vals[:initial_edge_idx],
            bt=bt,
            db=mass_balance_change
        )
        
        print(f"\n=== Equilibrium Length Change Calculations ===")
        print(f"Mass balance change: {mass_balance_change:.4f} m/yr")
        print(f"Glacier area: {glacier_area/1e6:.2f} km²")
        print(f"L_eq change (calc_Leq): {leq_change:.1f} m")
        print(f"Linear model steady-state: {linear_model.steady_state_length_change(mass_balance_change):.1f} m")
        
        # Create a perturbation time series matching the linear model's time step
        t_linear = np.arange(0, simulation_time, linear_model.dt)
        mass_balance_perturbation_ts = np.full_like(t_linear, mass_balance_change)
        
        # Calculate the length change evolution (Lp)
        Lp = linear_model.calc_length_change_for_mass_balance(mass_balance_perturbation_ts)
        linear_length_evolution = initial_length + Lp
        linear_final_length = linear_length_evolution[-1]

        flowline_final_length = ds['edge'].values[-1]
        
        length_error = abs(flowline_final_length - linear_final_length)
        relative_error = length_error / initial_length if initial_length > 0 else 0
        
        self._create_comparison_figure(ds, t_linear, linear_length_evolution, warming, relative_error)

        assert relative_error < 0.10, (
            f"Flowline and linear models disagree by {relative_error*100:.1f}%"
        )

    def _create_comparison_figure(self, ds, t_linear, linear_length_evolution, warming, relative_error):
        """Create a new, focused QC figure for the validation test."""
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1])
        fig.suptitle(f'Flowline vs. Linear Model Validation (+{warming}°C Warming)', fontsize=16)

        linear_final_length = linear_length_evolution[-1]
        initial_len_km = ds['edge'].values[0] / 1000
        final_len_km = ds['edge'].values[-1] / 1000
        linear_len_km = linear_final_length / 1000

        # Plot 1: Length Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ds['time'].values, ds['edge'].values / 1000, 'r-', linewidth=2, label='Flowline Model')
        ax1.axhline(y=initial_len_km, color='k', linestyle=':', label=f'Initial SS ({initial_len_km:.2f} km)')
        ax1.plot(t_linear, linear_length_evolution / 1000, color='orange', linestyle='--', label=f'Linear Model ({linear_len_km:.2f} km)')
        ax1.set_title('A) Glacier Length Evolution')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Length (km)')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # Plot 2: Length Difference (Error)
        ax2 = fig.add_subplot(gs[0, 1])
        flowline_length_interp = np.interp(t_linear, ds['time'].values, ds['edge'].values)
        length_diff = (flowline_length_interp - linear_length_evolution) / 1000  # in km
        ax2.plot(t_linear, length_diff, 'm-', label='Flowline - Linear')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.7)
        ax2.set_title('B) Length Difference (Error)')
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Difference (km)')
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        # Plot 3: Glacier Profiles and Thickness Change
        ax3 = fig.add_subplot(gs[1, 0])
        x_km = ds['x'].values / 1000
        edge_idx_initial = int(ds['edge_idx'].values[0])
        edge_idx_final = int(ds['edge_idx'].values[-1])
        
        h_initial = ds['h'].values[0, :]
        h_final = ds['h'].values[-1, :]
        s_initial = ds['zb'].values + h_initial
        s_final = ds['zb'].values + h_final

        ax3.plot(x_km, ds['zb'].values, 'k-', label='Bed')
        if edge_idx_initial > 0:
            ax3.plot(x_km[:edge_idx_initial], s_initial[:edge_idx_initial], 'b--', label='Initial Surface')
        if edge_idx_final > 0:
            ax3.plot(x_km[:edge_idx_final], s_final[:edge_idx_final], 'r-', label='Final Surface')
        
        # Shading for thickness change
        h_change = h_final - h_initial
        ax3.fill_between(x_km, s_final, s_initial, where=h_change < 0, color='red', alpha=0.3, interpolate=True, label='Thinning')
        ax3.fill_between(x_km, s_final, s_initial, where=h_change > 0, color='cyan', alpha=0.3, interpolate=True, label='Thickening')

        ax3.set_title('C) Glacier Profiles & Thickness Change')
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Elevation (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.5)

        # Plot 4: Mass Balance Profiles
        ax4 = fig.add_subplot(gs[1, 1])
        b_initial = ds['b_profile'].values[0, :]
        b_final = ds['b_profile'].values[-1, :]
        
        if edge_idx_initial > 0:
            ax4.plot(x_km[:edge_idx_initial], b_initial[:edge_idx_initial], 'b--', label='Initial MB')
        if edge_idx_final > 0:
            ax4.plot(x_km[:edge_idx_final], b_final[:edge_idx_final], 'r-', label='Final MB')
        
        ax4.axhline(0, color='k', linestyle=':', alpha=0.7)
        ax4.set_title('D) Mass Balance Profiles')
        ax4.set_xlabel('Distance (km)')
        ax4.set_ylabel('Mass Balance (m/yr)')
        ax4.legend()
        ax4.grid(True, alpha=0.5)

        # Plot 5: Summary Text
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Get diagnostic parameters for display
        spinup_file = QC_FIGURE_DIR / 'linear_model_validation_output' / "spinup_profiles" / "spinup_spinup_run_0000.nc"
        spinup_ds_diag = xr.open_dataset(spinup_file)
        initial_edge_idx_diag = int(spinup_ds_diag['edge_idx'].values[-1])
        ela_idx_diag = np.argmin(np.abs(spinup_ds_diag['x'].values - spinup_ds_diag['ela'].values[-1]))
        H_diag = np.mean(spinup_ds_diag['h'].values[-1, ela_idx_diag:initial_edge_idx_diag])
        bt_diag = spinup_ds_diag['b_profile'].values[-1, max(0, initial_edge_idx_diag - 1)]
        tau_diag = calculate_response_time(
            h=spinup_ds_diag['h'].values[-1, :],
            b=spinup_ds_diag['b_profile'].values[-1, :],
            delx=spinup_ds_diag.attrs['delx'],
            edge_idx=initial_edge_idx_diag,
            ela_idx=ela_idx_diag
        )
        
        summary_text = (
            f"Validation Summary: Warming Applied: +{warming}°C\n"
            f"Initial Length: {initial_len_km:.2f} km  |  H: {H_diag:.1f} m  |  bt: {bt_diag:.3f} m/yr  |  τ: {tau_diag:.1f} yr\n"
            f"Final Length (Flowline): {final_len_km:.2f} km\n"
            f"Final Length (Linear):   {linear_len_km:.2f} km\n"
            f"Absolute Difference: {abs(final_len_km - linear_len_km):.2f} km\n"
            f"Relative Error: {relative_error*100:.1f}% "
            f"-> {'PASSED' if relative_error < 0.10 else 'FAILED'}"
        )
        ax5.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        filename = 'flowline_linear_model_comparison.png'
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
        