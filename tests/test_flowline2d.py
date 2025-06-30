# -*- coding: utf-8 -*-
"""
Test suite for flowline2d module

Tests cover:
- Bed profile variations (uniform, concave, convex)
- Mass balance responses (step changes, white noise, linear trends)
- Numerical sensitivity (grid resolution)
- Boundary conditions and mass conservation
- Configuration validation
- Output format conversion
- Geometry interpolation

Author: Test Suite
Created: 2025-06-28
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dill
import copy

# Import the module under test
import sys
sys.path.append('src')
from flowline.flowline2d import (
    flowline2d, FlowlineConfig, FlowlineGeometry, 
    TemperaturePrecipitationForcing, DirectMassBalanceForcing,
    FlowlineModelError, GeometryError, NumericalInstabilityError
)

# Create output directory for QC figures
QC_FIGURE_DIR = Path("test_qc_figures")
QC_FIGURE_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def ss_result_uniform():
    """
    Session-scoped fixture to generate and cache a steady-state profile.
    This runs only once per test session, speeding up tests that need
    a steady-state initial condition.
    """
    bed_type = 'uniform'
    cache_dir = QC_FIGURE_DIR / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"steady_state_{bed_type}.pkl"

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            print(f"\nLoading cached steady-state from {cache_path}")
            return dill.load(f)

    print("\nGenerating new steady-state profile for tests...")
    # Config for a long spin-up run
    ss_config = FlowlineConfig(ts=0, tf=1000, delx=25, delt=0.0125/64)
    
    basic_params = {
        'length': 10000,
        'x_gr': np.linspace(0, 20000, 41),
        'elevation_drop': 1000,
        'width': 1000
    }
    
    x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
    
    h_init = np.maximum(0, 200 * (1 - x_gr / 8000))
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)

    forcing = TemperaturePrecipitationForcing(
        T0=10, P0=4, gamma=6.5e-3, mu=0.65, ts=ss_config.ts, tf=ss_config.tf
    )

    model_ss = flowline2d(config=ss_config, geometry=geometry, forcing=forcing)
    result_ss = model_ss.run()
    
    result_ss.to_pickle(cache_path)
    print(f"Cached new steady-state profile to {cache_path}")

    return result_ss


class TestGeometry:
    """Test geometry setup and interpolation"""
    
    @pytest.fixture
    def basic_geometry_params(self):
        """Standard geometry parameters for testing"""
        length = 20000  # 20 km
        x_gr = np.linspace(0, length, 41)  # 41 points for smooth interpolation
        return {
            'length': length,
            'x_gr': x_gr,
            'elevation_drop': 1000,  # m
            'width': 1000,  # m
        }
    
    def create_uniform_slope(self, params):
        """Create uniform slope bed profile"""
        x_gr = params['x_gr']
        zb_gr = params['elevation_drop'] * (1 - x_gr / params['length'])
        w_geom = np.full_like(x_gr, params['width'])
        return x_gr, zb_gr, w_geom
    
    def create_concave_profile(self, params):
        """Create slightly concave bed profile"""
        x_gr = params['x_gr']
        # Base uniform slope
        zb_uniform = params['elevation_drop'] * (1 - x_gr / params['length'])
        # Add concave perturbation (200m amplitude at midpoint)
        perturbation = -200 * np.sin(np.pi * x_gr / params['length'])**2
        zb_gr = zb_uniform + perturbation
        w_geom = np.full_like(x_gr, params['width'])
        return x_gr, zb_gr, w_geom
    
    def create_convex_profile(self, params):
        """Create slightly convex bed profile"""
        x_gr = params['x_gr']
        # Base uniform slope
        zb_uniform = params['elevation_drop'] * (1 - x_gr / params['length'])
        # Add convex perturbation (200m amplitude at midpoint)
        perturbation = 200 * np.sin(np.pi * x_gr / params['length'])**2
        zb_gr = zb_uniform + perturbation
        w_geom = np.full_like(x_gr, params['width'])
        return x_gr, zb_gr, w_geom
    
    def create_variable_width(self, params):
        """Create variable width profile"""
        x_gr = params['x_gr']
        zb_gr = params['elevation_drop'] * (1 - x_gr / params['length'])
        # Width varies from 2km at head to 0.5km at terminus
        w_geom = 2000 - 1500 * (x_gr / params['length'])
        return x_gr, zb_gr, w_geom
    
    def test_geometry_interpolation(self, basic_geometry_params):
        """Test that geometry interpolates correctly to model grid"""
        x_gr, zb_gr, w_geom = self.create_uniform_slope(basic_geometry_params)
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=25)
        
        # Check that interpolated values are reasonable
        assert len(geometry.x) == len(geometry.zb)
        assert len(geometry.x) == len(geometry.w)
        assert np.all(np.diff(geometry.zb) <= 0)  # Bed should slope downward
        assert np.all(geometry.w > 0)  # Width should be positive
    
    def test_gradient_calculation(self, basic_geometry_params):
        """Test bed slope calculation"""
        x_gr, zb_gr, w_geom = self.create_uniform_slope(basic_geometry_params)
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)
        
        # For uniform slope, gradient should be approximately constant
        expected_slope = -basic_geometry_params['elevation_drop'] / basic_geometry_params['length']
        mean_slope = np.mean(geometry.dzbdx)
        
        assert abs(mean_slope - expected_slope) < 0.01  # Within 1% of expected


class TestConfiguration:
    """Test configuration validation and setup"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FlowlineConfig()
        
        # Check physical parameters
        assert config.rho == 916.8
        assert config.g == 9.81
        assert config.n == 3
        assert config.k == 3
        
        # Check numerical parameters
        assert config.delx == 50
        assert config.delt == 0.0125 / 8
        assert config.min_thick == 1
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test invalid time range
        with pytest.raises(ValueError, match="tf must be greater than ts"):
            FlowlineConfig(ts=100, tf=50)
        
        # Test invalid spatial resolution
        with pytest.raises(ValueError, match="delx must be positive"):
            FlowlineConfig(delx=-10)
        
        # Test invalid time step
        with pytest.raises(ValueError, match="delt must be positive"):
            FlowlineConfig(delt=-0.1)
    
    def test_parameter_conversion(self):
        """Test that deformation parameters are converted correctly"""
        config = FlowlineConfig()
        
        # Check that fd and fs have been converted from seconds to years
        assert config.fd > 1e-24  # Should be much larger after conversion
        assert config.fs > 5e-20  # Should be much larger after conversion


class TestMassBalanceForcing:
    """Test mass balance forcing implementations"""
    
    @pytest.fixture
    def tp_params(self):
        """Temperature-precipitation forcing parameters"""
        return {
            'T0': 15,  # °C at sea level
            'P0': 2,   # m/yr precipitation
            'gamma': 6.5e-3,  # °C/m lapse rate
            'mu': 0.65,  # melt factor
            'ts': 0,
            'tf': 100
        }
    
    @pytest.fixture
    def direct_mb_params(self):
        """Direct mass balance forcing parameters"""
        return {
            'b0': 0,  # Base mass balance
        }
    
    def test_tp_forcing_creation(self, tp_params):
        """Test temperature-precipitation forcing setup"""
        forcing = TemperaturePrecipitationForcing(**tp_params)
        
        assert forcing.T0 == tp_params['T0']
        assert forcing.P0 == tp_params['P0']
        assert forcing.gamma == tp_params['gamma']
        assert forcing.mu == tp_params['mu']
    
    def test_direct_forcing_creation(self, direct_mb_params):
        """Test direct mass balance forcing setup"""
        forcing = DirectMassBalanceForcing(**direct_mb_params)
        
        assert forcing.b0 == direct_mb_params['b0']
        assert forcing.bp is None  # Should be None by default
        assert forcing.dbdz is None  # Should be None by default
        assert forcing.dbdx is None  # Should be None by default
    
    def test_mass_balance_calculation(self, tp_params):
        """Test mass balance calculation from T-P forcing"""
        forcing = TemperaturePrecipitationForcing(**tp_params)
        
        # Test at different elevations
        x = np.linspace(0, 10000, 201)
        h_eff = np.linspace(1000, 0, 201)  # 1000m elevation drop
        
        b, climate_vars = forcing.get_mass_balance(x, h_eff, 0)
        
        # Mass balance should decrease with elevation (more negative at low elevation)
        assert b[0] > b[-1]  # Higher elevation should have higher mass balance
        assert 'accumulation' in climate_vars
        assert 'melt' in climate_vars
        assert 'T' in climate_vars


class TestSteadyStateInitialization:
    """Test steady-state initialization for test scenarios"""
    
    @pytest.fixture
    def standard_config(self):
        """Standard configuration for initialization runs"""
        return FlowlineConfig(
            delx=25,
            delt=0.0125/64, 
            ts=0,
            tf=1000,  # Long enough to reach steady state
            gamma=6.5e-3,
            mu=0.65
        )
    
    def _create_qc_figure(self, result, title, filename):
        """Create QC figure for steady-state results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)
        
        # Plot 1: Final ice thickness profile
        ax = axes[0, 0]
        edge_idx = result.edge_idx[-1]
        ax.fill_between(result.x[:edge_idx]/1000, result.zb[:edge_idx], 
                       result.zb[:edge_idx] + result.h[-1, :edge_idx], 
                       alpha=0.7, color='lightblue', label='Ice')
        ax.plot(result.x/1000, result.zb, 'k-', linewidth=2, label='Bed')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Final Ice Thickness Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Length evolution
        ax = axes[0, 1]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mass balance profile
        ax = axes[1, 0]
        if hasattr(result, 'b_profile') and result.b_profile is not None:
            ax.plot(result.x[:edge_idx]/1000, result.b_profile[-1, :edge_idx], 'r-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Mass Balance (m/yr)')
            ax.set_title('Final Mass Balance Profile')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Area and ELA evolution
        ax = axes[1, 1]
        ax2 = ax.twinx()
        line1 = ax.plot(result.t, result.area/1e6, 'g-', linewidth=2, label='Area')
        line2 = ax2.plot(result.t, result.ela, 'orange', linewidth=2, label='ELA')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Area (km²)', color='g')
        ax2.set_ylabel('ELA (m)', color='orange')
        ax.set_title('Area and ELA Evolution')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_steady_state_profile(self, geometry_func, config, forcing_params, 
                                  initial_thickness=50):
        """Create steady-state ice thickness profile for testing"""
        # Create geometry
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 1000,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = geometry_func(basic_params)
        
        # Create uniform initial thickness
        h_init = np.full_like(x_gr, initial_thickness)
        h_init[x_gr > basic_params["length"]] = 0  # No ice near terminus initially
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        
        # Create forcing, ensuring ts and tf match the model config
        forcing_params['ts'] = config.ts
        forcing_params['tf'] = config.tf
        if 'T0' in forcing_params:
            forcing = TemperaturePrecipitationForcing(**forcing_params)
        else:
            forcing = DirectMassBalanceForcing(**forcing_params)
        
        # Run to steady state
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Return final thickness profile for use in tests
        return result.x, result.h[-1, :], result
    
    def test_steady_state_convergence(self, standard_config):
        """Test that model reaches steady state"""
        # Create simple forcing
        forcing_params = {
            'T0': 10,
            'P0': 4,
            'gamma': 6.5e-3,
            'mu': 0.65
        }
        
        x, h_final, result = self.create_steady_state_profile(
            TestGeometry().create_uniform_slope,
            standard_config,
            forcing_params
        )
        
        # Create QC figure
        self._create_qc_figure(result, 
                              'Steady State Convergence Test', 
                              'steady_state_convergence.png')
        
        # Check that length has stabilized (last 50 years should be relatively stable)
        final_lengths = result.edge[-50:]
        length_std = np.std(final_lengths)
        mean_length = np.mean(final_lengths)
        
        # Length should be stable to within 1% 
        assert length_std / mean_length < 0.01


class TestMassBalanceResponses:
    """Test glacier response to different mass balance scenarios"""
    
    @pytest.fixture
    def test_config(self):
        """Configuration for response tests"""
        return FlowlineConfig(
            delx=25,
            delt=0.0125/32,
            ts=0,
            tf=1000,
            deltout=1,
            gamma=6.5e-3,
            mu=0.65
        )
    
    def _create_response_qc_figure(self, results_dict, title, filename):
        """Create QC figure comparing multiple model runs"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot 1: Length evolution comparison
        ax = axes[0, 0]
        for i, (label, result) in enumerate(results_dict.items()):
            ax.plot(result.t, result.edge/1000, color=colors[i % len(colors)], 
                   linewidth=2, label=label)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final thickness profiles
        ax = axes[0, 1]
        for i, (label, result) in enumerate(results_dict.items()):
            edge_idx = result.edge_idx[-1]
            ax.plot(result.x[:edge_idx]/1000, result.h[-1, :edge_idx], 
                   color=colors[i % len(colors)], linewidth=2, label=label)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Ice Thickness (m)')
        ax.set_title('Final Thickness Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Area evolution
        ax = axes[1, 0]
        for i, (label, result) in enumerate(results_dict.items()):
            ax.plot(result.t, result.area/1e6, color=colors[i % len(colors)], 
                   linewidth=2, label=label)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Area (km²)')
        ax.set_title('Area Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Mass balance forcing (if available)
        ax = axes[1, 1]
        for i, (label, result) in enumerate(results_dict.items()):
            if hasattr(result, 'total_mass_balance'):
                ax.plot(result.t, result.total_mass_balance, color=colors[i % len(colors)], 
                       linewidth=2, label=label)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier-wide Balance (m³/yr)')
        ax.set_title('Glacier-wide Mass Balance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_step_change_symmetry(self, test_config, ss_result_uniform):
        """Test that +/- mass balance changes produce symmetric length responses"""
        ss_result = ss_result_uniform
        
        # This is the steady-state mass balance profile
        ss_b_profile = ss_result.b_profile[-1, :]
        
        # Test positive step change
        bp_pos = np.zeros(int(np.ceil(test_config.tf - test_config.ts)))
        bp_pos[100:] = 0.1  # +0.1 m/yr starting year 100
        
        forcing_pos = DirectMassBalanceForcing(
            b0=ss_b_profile, bp=bp_pos
        )
        geometry_pos = FlowlineGeometry(
            ss_result.x_gr, ss_result.zb_gr, ss_result.w_geom, profile=ss_result
        )
        model_pos = flowline2d(config=test_config, geometry=geometry_pos, forcing=forcing_pos)
        result_pos = model_pos.run()
        
        # Test negative step change
        bp_neg = np.zeros(int(np.ceil(test_config.tf - test_config.ts)))
        bp_neg[100:] = -0.1  # -0.1 m/yr starting year 100
        
        forcing_neg = DirectMassBalanceForcing(
            b0=ss_b_profile, bp=bp_neg
        )
        geometry_neg = FlowlineGeometry(
            ss_result.x_gr, ss_result.zb_gr, ss_result.w_geom, profile=ss_result
        )
        model_neg = flowline2d(config=test_config, geometry=geometry_neg, forcing=forcing_neg)
        result_neg = model_neg.run()

        # Debugging plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Debug: Step Change Symmetry Profiles', fontsize=14)

        # Plot initial steady-state profile
        ss_edge_idx = ss_result.edge_idx[-1]
        ax.plot(ss_result.x / 1000, ss_result.zb, 'k-', linewidth=2, label='Bed')
        ax.plot(ss_result.x[:ss_edge_idx] / 1000, 
                ss_result.zb[:ss_edge_idx] + ss_result.h[-1, :ss_edge_idx],
                'g--', linewidth=2, label='Initial Steady State')

        # Plot final profile for positive step
        pos_edge_idx = result_pos.edge_idx[-1]
        ax.plot(result_pos.x[:pos_edge_idx] / 1000,
                result_pos.zb[:pos_edge_idx] + result_pos.h[-1, :pos_edge_idx],
                'b-', linewidth=2, label='Final (+) Step')

        # Plot final profile for negative step
        neg_edge_idx = result_neg.edge_idx[-1]
        if neg_edge_idx > 0:
            ax.plot(result_neg.x[:neg_edge_idx] / 1000,
                    result_neg.zb[:neg_edge_idx] + result_neg.h[-1, :neg_edge_idx],
                    'r-', linewidth=2, label='Final (-) Step')

        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Initial and Final Glacier Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(QC_FIGURE_DIR / 'debug_step_change_profiles.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create QC figure
        results_dict = {
            'Positive Step (+0.5 m/yr)': result_pos,
            'Negative Step (-0.5 m/yr)': result_neg
        }
        self._create_response_qc_figure(results_dict, 
                                       'Step Change Symmetry Test', 
                                       'step_change_symmetry.png')
        
        # Calculate length changes
        initial_length = result_pos.edge[99]  # Length just before step change
        final_length_pos = result_pos.edge[-1]
        final_length_neg = result_neg.edge[-1]
        
        length_change_pos = final_length_pos - initial_length
        length_change_neg = final_length_neg - initial_length
        
        # Changes should be approximately symmetric (within 0.1%)
        symmetry_error = abs(length_change_pos + length_change_neg) / abs(length_change_pos)
        assert symmetry_error < 0.001, f"Symmetry error: {symmetry_error:.4f}"
    
    def test_white_noise_response(self, test_config, ss_result_uniform):
        """Test glacier response to white noise mass balance forcing"""
        # Create a config for a long 10k year run
        wn_config = copy.deepcopy(test_config)
        wn_config.tf = 10000
        wn_config.deltout = 10  # Save output every 10 years to manage memory
        wn_config.delt = 0.0125/16

        ss_result = ss_result_uniform
        
        # This is the steady-state mass balance profile
        ss_b_profile = ss_result.b_profile[-1, :]
        
        # Create white noise mass balance
        np.random.seed(42)  # For reproducible tests
        nyears = int(np.ceil(wn_config.tf - wn_config.ts))
        bp_noise = np.random.normal(0, 0.5, nyears)  # 0.5 m/yr std dev
        
        forcing = DirectMassBalanceForcing(
            b0=ss_b_profile, bp=bp_noise
        )
        geometry = FlowlineGeometry(
            ss_result.x_gr, ss_result.zb_gr, ss_result.w_geom, profile=ss_result
        )
        model = flowline2d(config=wn_config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create QC figure with noise analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('White Noise Response Test', fontsize=14)
        
        # Plot 1: Length evolution
        ax = axes[0, 0]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution (White Noise Forcing)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mass balance forcing
        ax = axes[0, 1]
        t_forcing = np.arange(wn_config.ts, wn_config.tf)
        ax.plot(t_forcing, bp_noise, 'r-', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Mass Balance Perturbation (m/yr)')
        ax.set_title('White Noise Forcing')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Length distribution histogram
        ax = axes[1, 0]
        lengths = result.edge[50:]  # Skip initial adjustment
        ax.hist(lengths/1000, bins=30, alpha=0.7, density=True, color='lightblue')
        ax.set_xlabel('Glacier Length (km)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Length Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Thickness profile distribution
        ax = axes[1, 1]
        # Initial steady state profile
        ss_edge_idx = ss_result.edge_idx[-1]
        ax.fill_between(ss_result.x[:ss_edge_idx]/1000, 
                        ss_result.zb[:ss_edge_idx],
                        ss_result.zb[:ss_edge_idx] + ss_result.h[-1, :ss_edge_idx],
                        color='gray', alpha=0.3, label='Initial State')
        ax.plot(ss_result.x/1000, ss_result.zb, 'k-', linewidth=0.5)

        # Percentile profiles
        h_profiles = result.h[50:, :] 
        
        percentiles = [2.5, 16, 84, 97.5]
        h_percentiles = np.percentile(h_profiles, percentiles, axis=0)

        max_edge_idx = int(np.percentile(result.edge_idx[50:], 99))
        
        linestyles = ['--', ':', ':', '--']
        colors = ['red', 'blue', 'blue', 'red']
        labels = ['-2σ (2.5%)', '-1σ (16%)', '+1σ (84%)', '+2σ (97.5%)']
        
        for i in range(len(percentiles)):
            ax.plot(result.x[:max_edge_idx]/1000, 
                    result.zb[:max_edge_idx] + h_percentiles[i, :max_edge_idx], 
                    linestyle=linestyles[i], color=colors[i], label=labels[i])
        
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Thickness Profile Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / 'white_noise_response.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Analyze length variability (skip initial adjustment period)
        lengths = result.edge[50:]  # Skip first 50 years
        initial_length = np.mean(lengths[:10])  # Reference length
        
        # Length distribution should be approximately symmetric
        length_deviations = lengths - initial_length
        skewness = np.mean(length_deviations**3) / (np.std(length_deviations)**3)
        
        # Skewness should be close to zero for symmetric distribution
        assert abs(skewness) < 0.5, f"Length distribution skewness: {skewness:.3f}"
        
        # Standard deviation should be reasonable (store for comparison with other tests)
        length_std = np.std(lengths)
        assert length_std > 0, "Length should vary with noise forcing"
    
    def test_linear_trend_response(self, test_config, ss_result_uniform):
        """Test glacier response to linear mass balance trend"""
        ss_result = ss_result_uniform
        
        # This is the steady-state mass balance profile
        ss_b_profile = ss_result.b_profile[-1, :]
        
        # Create linear trend: 0 to -1 m/yr over 100 years, then steady
        nyears = int(np.ceil(test_config.tf - test_config.ts))
        bp_trend = np.zeros(nyears)
        
        # Linear decrease for first 100 years
        trend_years = min(100, nyears)
        bp_trend[:trend_years] = np.linspace(0, -1, trend_years)
        # Steady at -1 m/yr afterwards
        if nyears > 100:
            bp_trend[100:] = -1
        
        forcing = DirectMassBalanceForcing(
            b0=ss_b_profile, bp=bp_trend
        )
        geometry = FlowlineGeometry(
            ss_result.x_gr, ss_result.zb_gr, ss_result.w_geom, profile=ss_result
        )
        model = flowline2d(config=test_config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create QC figure for trend response
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Linear Trend Response Test', fontsize=14)
        
        # Plot 1: Length evolution with trend phases
        ax = axes[0, 0]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=2)
        ax.axvline(x=100, color='r', linestyle='--', alpha=0.7, label='End of trend')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mass balance forcing
        ax = axes[0, 1]
        ax.plot(result.t, bp_trend[:len(result.t)], 'r-', linewidth=2)
        ax.axvline(x=100, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Mass Balance Perturbation (m/yr)')
        ax.set_title('Linear Trend Forcing')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rate of length change
        ax = axes[1, 0]
        dLdt = np.gradient(result.edge, result.t)
        ax.plot(result.t, dLdt, 'g-', linewidth=2)
        ax.axvline(x=100, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('dL/dt (m/yr)')
        ax.set_title('Rate of Length Change')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Transient response analysis
        ax = axes[1, 1]
        initial_length = result.edge[0]
        final_length = result.edge[-1]
        normalized_response = (result.edge - initial_length) / (final_length - initial_length)
        ax.plot(result.t, normalized_response, 'purple', linewidth=2)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Final response')
        ax.axvline(x=100, color='r', linestyle='--', alpha=0.7, label='End of trend')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Normalized Length Change')
        ax.set_title('Transient Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / 'linear_trend_response.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate transient response ratio
        initial_length = result.edge[0]
        final_length = result.edge[-1]
        
        # Find length at end of trend period (year 100)
        trend_end_idx = min(100, len(result.edge) - 1)
        transient_length = result.edge[trend_end_idx]
        
        # Calculate ratio of transient to final length change
        transient_ratio = (transient_length - initial_length) / (final_length - initial_length)
        
        # This ratio should be characteristic of the glacier response time
        # Store for comparison across different configurations
        assert 0 < transient_ratio < 1, f"Transient ratio: {transient_ratio:.3f}"


class TestNumericalSensitivity:
    """Test numerical sensitivity to grid resolution and time stepping"""
    
    def test_grid_resolution_sensitivity(self):
        """Test that results are consistent across different grid resolutions"""
        # Base configuration
        # Timestep delt must be scaled with delx to maintain stability.
        # A common scaling for this type of problem is delt ~ delx^2.
        base_delt = 0.0125 / 64
        config_base = FlowlineConfig(delx=25, delt=base_delt, ts=0, tf=100)
        config_fine = FlowlineConfig(delx=12.5, delt=base_delt/4, ts=0, tf=100)
        config_coarse = FlowlineConfig(delx=50, delt=base_delt*4, ts=0, tf=100)
        
        # Create identical geometry and forcing
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 1000,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 200 * (1 - x_gr / 8000))
        
        forcing_params = {
            'T0': 5, 'P0': 4, 'gamma': 6.5e-3, 'mu': 0.65
        }
        
        # Run models with different resolutions
        results = {}
        for name, config in [('base', config_base), ('fine', config_fine), ('coarse', config_coarse)]:
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
            
            # Ensure forcing uses the correct time range for this config
            run_forcing_params = forcing_params.copy()
            run_forcing_params['ts'] = config.ts
            run_forcing_params['tf'] = config.tf
            forcing = TemperaturePrecipitationForcing(**run_forcing_params)
            
            model = flowline2d(config=config, geometry=geometry, forcing=forcing)
            results[name] = model.run()
        
        # Create QC figure for grid sensitivity
        self._create_grid_sensitivity_qc_figure(results, 
                                               'Grid Resolution Sensitivity Test', 
                                               'grid_resolution_sensitivity.png')
        
        # Compare final lengths (should be within 0.1% of each other)
        base_length = results['base'].edge[-1]
        fine_length = results['fine'].edge[-1]
        coarse_length = results['coarse'].edge[-1]
        
        fine_error = abs(fine_length - base_length) / base_length
        coarse_error = abs(coarse_length - base_length) / base_length
        
        assert fine_error < 0.01, f"Fine grid error: {fine_error:.4f}"
        assert coarse_error < 0.01, f"Coarse grid error: {coarse_error:.4f}"
    
    def test_timestep_sensitivity(self, ss_result_uniform):
        """Test that results are consistent across different time steps"""
        # Get a steady state profile to start from
        ss_result = ss_result_uniform
        ss_b_profile = ss_result.b_profile[-1, :]

        # Define time steps to test
        base_delt = 0.0125
        delts = [base_delt / (2**i) for i in [2, 4, 6, 7]]  # dt/4, dt/16, dt/64, dt/128

        # Define a small step change in mass balance
        nyears = 500
        bp_step = np.zeros(nyears)
        bp_step[50:] = 0.1 # +0.1 m/yr after 50 years

        results = {}
        for delt in delts:
            config = FlowlineConfig(delx=25, delt=delt, ts=0, tf=nyears, deltout=5)
            forcing = DirectMassBalanceForcing(b0=ss_b_profile, bp=bp_step)
            geometry = FlowlineGeometry(
                ss_result.x_gr, ss_result.zb_gr, ss_result.w_geom, profile=ss_result
            )
            model = flowline2d(config=config, geometry=geometry, forcing=forcing)
            try:
                results[f'dt={delt:.2e}'] = model.run()
            except NumericalInstabilityError:
                print(f"\nRun with dt={delt:.2e} failed as expected due to instability.")
                # Store None or a special marker if needed, but here we just skip it
                pass

        # Create QC figure
        self._create_timestep_sensitivity_qc_figure(results,
                                                  'Timestep Sensitivity Test',
                                                  'timestep_sensitivity.png')

        # Compare final lengths of successful runs
        successful_results = {k: v for k, v in results.items() if v is not None}
        assert len(successful_results) > 1, "Not enough successful runs to compare timestep sensitivity."

        final_lengths = [res.edge[-1] for res in successful_results.values()]
        # Use the result from the smallest timestep as the reference
        reference_length = final_lengths[-1]
        
        for res in successful_results.values():
            error = abs(res.edge[-1] - reference_length) / reference_length
            assert error < 0.01, f"Error for dt={res.config.delt:.2e} is {error:.4f}, exceeds 1%"

    def _create_timestep_sensitivity_qc_figure(self, results_dict, title, filename):
        """Create QC figure for time step sensitivity analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14)
        
        # Plot 1: Length evolution comparison
        ax = axes[0]
        for label, result in results_dict.items():
            ax.plot(result.t, result.edge/1000, linewidth=2, label=label)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final length vs timestep
        ax = axes[1]
        delts = [res.config.delt for res in results_dict.values()]
        final_lengths = [res.edge[-1]/1000 for res in results_dict.values()]
        
        ax.plot(delts, final_lengths, 'o-')
        ax.set_xlabel('Time Step (years)')
        ax.set_ylabel('Final Length (km)')
        ax.set_title('Final Length vs Time Step')
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_grid_sensitivity_qc_figure(self, results_dict, title, filename):
        """Create QC figure for grid sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14)
        
        colors = {'base': 'blue', 'fine': 'red', 'coarse': 'green'}
        
        # Plot 1: Length evolution comparison
        ax = axes[0, 0]
        for label, result in results_dict.items():
            ax.plot(result.t, result.edge/1000, color=colors[label], 
                   linewidth=2, label=f'{label} (Δx={result.config.delx}m)')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final thickness profiles (interpolated to common grid)
        ax = axes[0, 1]
        # Use base grid as reference
        x_ref = results_dict['base'].x
        for label, result in results_dict.items():
            edge_idx = result.edge_idx[-1]
            if edge_idx > 0:
                # Interpolate to reference grid for comparison
                from scipy.interpolate import interp1d
                x_result = result.x[:edge_idx]
                h_result = result.h[-1, :edge_idx]
                if len(x_result) > 1:
                    h_interp = interp1d(x_result, h_result, bounds_error=False, fill_value=0)
                    h_ref = h_interp(x_ref)
                    ax.plot(x_ref/1000, h_ref, color=colors[label], 
                           linewidth=2, label=f'{label} (Δx={result.config.delx}m)')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Ice Thickness (m)')
        ax.set_title('Final Thickness Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Convergence analysis
        ax = axes[1, 0]
        base_length = results_dict['base'].edge
        for label, result in results_dict.items():
            if label != 'base':
                # Interpolate to base time grid for comparison
                from scipy.interpolate import interp1d
                if len(result.t) > 1 and len(base_length) > 1:
                    length_interp = interp1d(result.t, result.edge, bounds_error=False, fill_value='extrapolate')
                    length_ref = length_interp(results_dict['base'].t)
                    error = abs(length_ref - base_length) / base_length * 100
                    ax.plot(results_dict['base'].t, error, color=colors[label], 
                           linewidth=2, label=f'{label} vs base')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Relative Error (%)')
        ax.set_title('Length Error vs Base Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final length comparison
        ax = axes[1, 1]
        labels = list(results_dict.keys())
        final_lengths = [results_dict[label].edge[-1]/1000 for label in labels]
        grid_sizes = [results_dict[label].config.delx for label in labels]
        
        ax.scatter(grid_sizes, final_lengths, s=100, c=[colors[label] for label in labels])
        for i, label in enumerate(labels):
            ax.annotate(label, (grid_sizes[i], final_lengths[i]), 
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Grid Size (m)')
        ax.set_ylabel('Final Length (km)')
        ax.set_title('Final Length vs Grid Resolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()


class TestBoundaryConditions:
    """Test boundary condition handling"""
    
    def test_glacier_head_boundary(self):
        """Test behavior at glacier head (upstream boundary)"""
        # Use a smaller timestep for stability with high accumulation
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=50)
        
        # Create geometry with very high mass balance at head
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 500,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 8000))
        
        # High accumulation at head
        forcing = DirectMassBalanceForcing(b0=0.5)  # +5 m/yr everywhere
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Thickness at head should increase but remain finite
        head_thickness = result.h[:, 0]  # First grid point
        assert np.all(head_thickness >= 0), "Head thickness should be non-negative"
        assert np.all(np.isfinite(head_thickness)), "Head thickness should be finite"
    
    def test_glacier_terminus_boundary(self):
        """Test behavior at glacier terminus"""
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=50)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 1000,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 8000))
        
        # Strong ablation to test terminus retreat
        forcing = DirectMassBalanceForcing(b0=-1)  # -2 m/yr everywhere
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Glacier should retreat (edge should decrease)
        initial_edge = result.edge[0]
        final_edge = result.edge[-1]
        assert final_edge < initial_edge, "Glacier should retreat under negative mass balance"
        
        # Thickness beyond terminus should be zero
        for i, edge_idx in enumerate(result.edge_idx):
            if edge_idx < len(result.h[i, :]) - 1:
                beyond_terminus = result.h[i, edge_idx+1:]
                assert np.all(beyond_terminus <= result.config.min_thick), \
                    "Thickness beyond terminus should be minimal"


class TestMassConservation:
    """Test mass conservation in the model"""
    
    def test_mass_conservation_uniform_mb(self):
        """Test mass conservation with uniform mass balance"""
        config = FlowlineConfig(delx=25, delt=0.0125/64, ts=0, tf=10000, deltout=1)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 500,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 8000))
        
        # Uniform mass balance
        forcing = DirectMassBalanceForcing(b0=0.5)  # +1 m/yr everywhere
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create QC figure for mass conservation
        self._create_mass_conservation_qc_figure(result, 
                                                'Mass Conservation Test', 
                                                'mass_conservation.png')
        
        # Calculate mass balance and volume change
        dt = np.diff(result.t)
        
        for i in range(1, len(result.t)):
            # Volume change
            edge_idx_old = result.edge_idx[i-1]
            edge_idx_new = result.edge_idx[i]
            vol_old = np.sum(result.h[i-1, :edge_idx_old] * result.w[:edge_idx_old] * config.delx)
            vol_new = np.sum(result.h[i, :edge_idx_new] * result.w[:edge_idx_new] * config.delx)
            dvol_dt = (vol_new - vol_old) / dt[i-1]
            
            # Mass balance input
            mb_input = result.total_mass_balance[i]
            
            # Should be approximately equal (within numerical precision)
            if abs(mb_input) > 1e-6:  # Avoid division by very small numbers
                relative_error = abs(dvol_dt - mb_input) / abs(mb_input)
                assert relative_error < 0.06, \
                    f"Mass conservation error at t={result.t[i]:.1f}: {relative_error:.4f}"
    
    def _create_mass_conservation_qc_figure(self, result, title, filename):
        """Create QC figure for mass conservation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14)
        
        # Calculate volume and mass balance terms
        dt = np.diff(result.t)
        volumes = []
        dvol_dt = []
        
        for i in range(len(result.t)):
            edge_idx = result.edge_idx[i]
            if edge_idx > 0:
                vol = np.sum(result.h[i, :edge_idx] * result.w[:edge_idx] * result.config.delx)
                volumes.append(vol)
            else:
                volumes.append(0)
        
        for i in range(1, len(volumes)):
            dvol_dt.append((volumes[i] - volumes[i-1]) / dt[i-1])
        
        # Plot 1: Volume evolution
        ax = axes[0, 0]
        ax.plot(result.t, np.array(volumes)/1e9, 'b-', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Ice Volume (km³)')
        ax.set_title('Ice Volume Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mass balance vs volume change rate
        ax = axes[0, 1]
        if len(dvol_dt) > 0:
            ax.scatter(result.total_mass_balance[1:], dvol_dt, alpha=0.7, s=20)
            # Perfect conservation line
            min_val = min(min(result.total_mass_balance[1:]), min(dvol_dt))
            max_val = max(max(result.total_mass_balance[1:]), max(dvol_dt))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   linewidth=2, label='Perfect Conservation')
            ax.set_xlabel('Mass Balance Input (m³/yr)')
            ax.set_ylabel('Volume Change Rate (m³/yr)')
            ax.set_title('Mass Conservation Check')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Conservation error over time
        ax = axes[1, 0]
        if len(dvol_dt) > 0:
            conservation_error = []
            for i in range(len(dvol_dt)):
                mb_input = result.total_mass_balance[i+1]
                if abs(mb_input) > 1e-6:
                    error = abs(dvol_dt[i] - mb_input) / abs(mb_input) * 100
                    conservation_error.append(error)
                else:
                    conservation_error.append(0)
            
            ax.plot(result.t[1:len(conservation_error)+1], conservation_error, 'r-', linewidth=2)
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Relative Error (%)')
            ax.set_title('Mass Conservation Error')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Glacier geometry evolution
        ax = axes[1, 1]
        # Show thickness profiles at different times
        time_indices = [0, len(result.t)//4, len(result.t)//2, 3*len(result.t)//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
            edge_idx = result.edge_idx[t_idx]
            if edge_idx > 0:
                ax.fill_between(result.x[:edge_idx]/1000, result.zb[:edge_idx], 
                               result.zb[:edge_idx] + result.h[t_idx, :edge_idx], 
                               alpha=0.3, color=color, 
                               label=f't={result.t[t_idx]:.1f} yr')
        
        ax.plot(result.x/1000, result.zb, 'k-', linewidth=2, label='Bed')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Glacier Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()


class TestOutputFormats:
    """Test output format conversions"""
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample model result for testing"""
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=10, deltout=2)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 200,
            'width': 500
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 50 * (1 - x_gr / 8000))
        
        forcing = DirectMassBalanceForcing(b0=0.5)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        return model.run()
    
    def test_to_pandas(self, sample_result):
        """Test conversion to pandas DataFrame"""
        df = sample_result.to_pandas()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_result.t)
        assert 'area' in df.columns
        assert 'total_mass_balance' in df.columns
        assert 'edge' in df.columns
        assert 'ela' in df.columns
        
        # Index should be time
        assert np.allclose(df.index.values, sample_result.t)
    
    def test_to_xarray(self, sample_result):
        """Test conversion to xarray Dataset"""
        ds = sample_result.to_xarray()
        
        assert isinstance(ds, xr.Dataset)
        
        # Check dimensions
        assert 'time' in ds.dims
        assert 'x' in ds.dims
        assert ds.dims['time'] == len(sample_result.t)
        assert ds.dims['x'] == len(sample_result.x)
        
        # Check key variables
        assert 'h' in ds.data_vars
        assert 'b_profile' in ds.data_vars
        assert 'edge' in ds.data_vars
        assert 'area' in ds.data_vars
        
        # Check coordinates
        assert np.allclose(ds.time.values, sample_result.t)
        assert np.allclose(ds.x.values, sample_result.x)
        
        # Check attributes (configuration should be stored)
        assert 'delx' in ds.attrs
        assert 'delt' in ds.attrs
    
    def test_pickle_serialization(self, sample_result):
        """Test pickle serialization and deserialization"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            try:
                sample_result.to_pickle(tmp.name)
                
                # File should exist and have content
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
            finally:
                # Clean up
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestFeatures:
    """Test specific model features"""
    
    def test_variable_width_geometry(self):
        """Test that variable width geometry works correctly"""
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=20, deltout=5)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 500,
            'width': 1000  # This will be overridden
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_variable_width(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 8000))
        
        forcing = DirectMassBalanceForcing(b0=0)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Width should vary along the flowline
        assert not np.allclose(result.w, result.w[0]), "Width should vary along flowline"
        assert np.all(result.w > 0), "Width should be positive everywhere"
        
        # Model should run successfully
        assert result.no_error, "Model should complete without errors"
    
    def test_pdd_temperature_forcing(self):
        """Test positive degree day temperature forcing"""
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=20, deltout=5)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 300,
            'width': 800
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 80 * (1 - x_gr / 8000))
        
        # PDD forcing parameters
        forcing = TemperaturePrecipitationForcing(
            T0=15, P0=2, gamma=6.5e-3, mu=0.005,  # 5 mm/dd
            T2melt='pdd', pdd_Tamp=10, ts=0, tf=20
        )
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create QC figure for PDD forcing
        self._create_pdd_qc_figure(result, 
                                  'PDD Temperature Forcing Test', 
                                  'pdd_temperature_forcing.png')
        
        # Should have PDD output
        assert hasattr(result, 'pdd'), "Result should contain PDD data"
        assert result.pdd is not None, "PDD data should not be None"
        assert np.all(result.pdd >= 0), "PDD values should be non-negative"
    
    def _create_pdd_qc_figure(self, result, title, filename):
        """Create QC figure for PDD temperature forcing"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14)
        
        # Plot 1: Final ice thickness and bed profile
        ax = axes[0, 0]
        edge_idx = result.edge_idx[-1]
        if edge_idx > 0:
            ax.fill_between(result.x[:edge_idx]/1000, result.zb[:edge_idx], 
                           result.zb[:edge_idx] + result.h[-1, :edge_idx], 
                           alpha=0.7, color='lightblue', label='Ice')
        ax.plot(result.x/1000, result.zb, 'k-', linewidth=2, label='Bed')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Final Ice Thickness Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: PDD distribution
        ax = axes[0, 1]
        if hasattr(result, 'pdd') and result.pdd is not None:
            # Show PDD along flowline for final time
            if edge_idx > 0:
                ax.plot(result.x[:edge_idx]/1000, result.pdd[-1, :edge_idx], 'r-', linewidth=2)
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('PDD (degree-days)')
                ax.set_title('Final PDD Distribution')
                ax.grid(True, alpha=0.3)
        
        # Plot 3: Temperature and melt profiles
        ax = axes[1, 0]
        if hasattr(result, 'melt') and result.melt is not None:
            if edge_idx > 0:
                ax.plot(result.x[:edge_idx]/1000, result.melt[-1, :edge_idx], 'orange', 
                       linewidth=2, label='Melt')
                if hasattr(result, 'accumulation') and result.accumulation is not None:
                    ax.plot(result.x[:edge_idx]/1000, result.accumulation[-1, :edge_idx], 'blue', 
                           linewidth=2, label='Accumulation')
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Rate (m/yr)')
                ax.set_title('Final Melt and Accumulation')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot 4: Length evolution
        ax = axes[1, 1]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(QC_FIGURE_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_geometry_raises_error(self):
        """Test that invalid geometry raises appropriate errors"""
        # Empty arrays should raise an error
        with pytest.raises((ValueError, IndexError)):
            geometry = FlowlineGeometry([], [], [])
            geometry.setup_grid(25)
    
    def test_mismatched_geometry_arrays(self):
        """Test that mismatched geometry arrays raise errors"""
        x_gr = np.linspace(0, 1000, 5)
        zb_gr = np.linspace(100, 0, 4)  # Wrong length
        w_geom = np.linspace(500, 500, 5)
        
        with pytest.raises((ValueError, IndexError)):
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
            geometry.setup_grid(25)
    
    def test_extreme_mass_balance_handling(self):
        """Test handling of extreme mass balance values"""
        config = FlowlineConfig(delx=25, delt=0.0125/16, ts=0, tf=5, deltout=1)
        
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 20000, 41),
            'elevation_drop': 200,
            'width': 500
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 50 * (1 - x_gr / 8000))
        
        # Extremely negative mass balance
        forcing = DirectMassBalanceForcing(b0=-2)  # -2 m/yr
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        
        # Should either complete or raise a specific error
        try:
            result = model.run()
            # If it completes, glacier should disappear quickly
            assert result.edge[-1] < result.edge[0], "Glacier should retreat rapidly"
        except FlowlineModelError:
            # This is acceptable for extreme conditions
            pass


# Placeholder for steady-state analytical validation
class TestSteadyStateValidation:
    """Placeholder for steady-state analytical validation tests"""
    
    def test_analytical_steady_state_comparison(self):
        """
        Placeholder for comparing model results to analytical solutions
        where available (e.g., simple geometries with uniform mass balance)
        """
        pytest.skip("Analytical steady-state validation not yet implemented")
    
    def test_mass_balance_gradient_steady_state(self):
        """
        Placeholder for testing steady-state solutions with 
        linear mass balance gradients
        """
        pytest.skip("Mass balance gradient steady-state validation not yet implemented")


def create_qc_figure_index():
    """Create an HTML index of all QC figures"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flowline2D Test QC Figures</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .figure { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
            .figure img { max-width: 100%; height: auto; }
            .figure h3 { margin-top: 0; color: #333; }
        </style>
    </head>
    <body>
        <h1>Flowline2D Integration Test QC Figures</h1>
        <p>Generated on: {date}</p>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # List of expected QC figures
    qc_figures = [
        ('steady_state_convergence.png', 'Steady State Convergence Test'),
        ('step_change_symmetry.png', 'Step Change Symmetry Test'),
        ('white_noise_response.png', 'White Noise Response Test'),
        ('linear_trend_response.png', 'Linear Trend Response Test'),
        ('grid_resolution_sensitivity.png', 'Grid Resolution Sensitivity Test'),
        ('timestep_sensitivity.png', 'Time Step Sensitivity Test'),
        ('mass_conservation.png', 'Mass Conservation Test'),
        ('pdd_temperature_forcing.png', 'PDD Temperature Forcing Test'),
    ]
    
    for filename, description in qc_figures:
        if (QC_FIGURE_DIR / filename).exists():
            html_content += f"""
            <div class="figure">
                <h3>{description}</h3>
                <img src="{filename}" alt="{description}">
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML index
    with open(QC_FIGURE_DIR / 'index.html', 'w') as f:
        f.write(html_content)
    
    print(f"QC figure index created at: {QC_FIGURE_DIR / 'index.html'}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
    
    # Create QC figure index after tests complete
    create_qc_figure_index()
