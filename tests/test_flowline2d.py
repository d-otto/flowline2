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

# Import the module under test
import sys
sys.path.append('src')
from flowline.flowline2d import (
    flowline2d, FlowlineConfig, FlowlineGeometry, 
    TemperaturePrecipitationForcing, DirectMassBalanceForcing,
    FlowlineModelError, GeometryError, NumericalInstabilityError
)


class TestGeometry:
    """Test geometry setup and interpolation"""
    
    @pytest.fixture
    def basic_geometry_params(self):
        """Standard geometry parameters for testing"""
        length = 10000  # 10 km
        x_gr = np.linspace(0, length, 21)  # 21 points for smooth interpolation
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
        geometry.setup_grid(delx=50)
        
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
            'ts': 0,
            'tf': 100
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
    
    def test_mass_balance_calculation(self, tp_params):
        """Test mass balance calculation from T-P forcing"""
        forcing = TemperaturePrecipitationForcing(**tp_params)
        
        # Test at different elevations
        x = np.linspace(0, 10000, 201)
        h_eff = np.linspace(1000, 0, 201)  # 1000m elevation drop
        
        b, climate_vars = forcing.get_mass_balance(x, h_eff, 0)
        
        # Mass balance should decrease with elevation (more negative at low elevation)
        assert b[0] > b[-1]  # Higher elevation should have higher mass balance
        assert 'P' in climate_vars
        assert 'melt' in climate_vars
        assert 'T' in climate_vars


class TestSteadyStateInitialization:
    """Test steady-state initialization for test scenarios"""
    
    @pytest.fixture
    def standard_config(self):
        """Standard configuration for initialization runs"""
        return FlowlineConfig(
            delx=50,
            delt=0.0125/8,
            ts=0,
            tf=500,  # Long enough to reach steady state
            deltout=10,  # Save every 10 years
            gamma=6.5e-3,
            mu=0.65
        )
    
    def create_steady_state_profile(self, geometry_func, config, forcing_params, 
                                  initial_thickness=100):
        """Create steady-state ice thickness profile for testing"""
        # Create geometry
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 10000, 21),
            'elevation_drop': 1000,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = geometry_func(basic_params)
        
        # Create uniform initial thickness
        h_init = np.full_like(x_gr, initial_thickness)
        h_init[x_gr > 8000] = 0  # No ice near terminus initially
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        
        # Create forcing
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
            'T0': 15,
            'P0': 2,
            'gamma': 6.5e-3,
            'mu': 0.65,
            'ts': 0,
            'tf': 500
        }
        
        x, h_final, result = self.create_steady_state_profile(
            TestGeometry().create_uniform_slope,
            standard_config,
            forcing_params
        )
        
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
            delx=50,
            delt=0.0125/4,  # Slightly larger time step for faster testing
            ts=0,
            tf=200,
            deltout=1,
            gamma=6.5e-3,
            mu=0.65
        )
    
    def get_steady_state_setup(self, bed_type='uniform'):
        """Get steady-state initial conditions"""
        # This would use pre-computed steady states in practice
        # For now, create reasonable initial conditions
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 10000, 21),
            'elevation_drop': 1000,
            'width': 1000
        }
        
        if bed_type == 'uniform':
            x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        elif bed_type == 'concave':
            x_gr, zb_gr, w_geom = TestGeometry().create_concave_profile(basic_params)
        elif bed_type == 'convex':
            x_gr, zb_gr, w_geom = TestGeometry().create_convex_profile(basic_params)
        
        # Create reasonable initial thickness (triangular profile)
        h_init = np.maximum(0, 200 * (1 - x_gr / 8000))
        
        return x_gr, zb_gr, w_geom, h_init
    
    def test_step_change_symmetry(self, test_config):
        """Test that +/- mass balance changes produce symmetric length responses"""
        x_gr, zb_gr, w_geom, h_init = self.get_steady_state_setup('uniform')
        
        # Test positive step change
        bp_pos = np.zeros(int(test_config.tf - test_config.ts))
        bp_pos[50:] = 0.5  # +0.5 m/yr after year 50
        
        forcing_pos = DirectMassBalanceForcing(
            b0=0, bp=bp_pos, ts=test_config.ts, tf=test_config.tf
        )
        geometry_pos = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model_pos = flowline2d(config=test_config, geometry=geometry_pos, forcing=forcing_pos)
        result_pos = model_pos.run()
        
        # Test negative step change
        bp_neg = np.zeros(int(test_config.tf - test_config.ts))
        bp_neg[50:] = -0.5  # -0.5 m/yr after year 50
        
        forcing_neg = DirectMassBalanceForcing(
            b0=0, bp=bp_neg, ts=test_config.ts, tf=test_config.tf
        )
        geometry_neg = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model_neg = flowline2d(config=test_config, geometry=geometry_neg, forcing=forcing_neg)
        result_neg = model_neg.run()
        
        # Calculate length changes
        initial_length = result_pos.edge[49]  # Length just before step change
        final_length_pos = result_pos.edge[-1]
        final_length_neg = result_neg.edge[-1]
        
        length_change_pos = final_length_pos - initial_length
        length_change_neg = final_length_neg - initial_length
        
        # Changes should be approximately symmetric (within 0.1%)
        symmetry_error = abs(length_change_pos + length_change_neg) / abs(length_change_pos)
        assert symmetry_error < 0.001, f"Symmetry error: {symmetry_error:.4f}"
    
    def test_white_noise_response(self, test_config):
        """Test glacier response to white noise mass balance forcing"""
        x_gr, zb_gr, w_geom, h_init = self.get_steady_state_setup('uniform')
        
        # Create white noise mass balance
        np.random.seed(42)  # For reproducible tests
        nyears = int(test_config.tf - test_config.ts)
        bp_noise = np.random.normal(0, 0.65, nyears)  # 0.65 m/yr std dev
        
        forcing = DirectMassBalanceForcing(
            b0=0, bp=bp_noise, ts=test_config.ts, tf=test_config.tf
        )
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=test_config, geometry=geometry, forcing=forcing)
        result = model.run()
        
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
    
    def test_linear_trend_response(self, test_config):
        """Test glacier response to linear mass balance trend"""
        x_gr, zb_gr, w_geom, h_init = self.get_steady_state_setup('uniform')
        
        # Create linear trend: 0 to -1 m/yr over 100 years, then steady
        nyears = int(test_config.tf - test_config.ts)
        bp_trend = np.zeros(nyears)
        
        # Linear decrease for first 100 years
        trend_years = min(100, nyears)
        bp_trend[:trend_years] = np.linspace(0, -1, trend_years)
        # Steady at -1 m/yr afterwards
        if nyears > 100:
            bp_trend[100:] = -1
        
        forcing = DirectMassBalanceForcing(
            b0=0, bp=bp_trend, ts=test_config.ts, tf=test_config.tf
        )
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=test_config, geometry=geometry, forcing=forcing)
        result = model.run()
        
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
        config_base = FlowlineConfig(delx=50, delt=0.0125/8, ts=0, tf=100, deltout=5)
        config_fine = FlowlineConfig(delx=25, delt=0.0125/8, ts=0, tf=100, deltout=5)
        config_coarse = FlowlineConfig(delx=100, delt=0.0125/8, ts=0, tf=100, deltout=5)
        
        # Create identical geometry and forcing
        basic_params = {
            'length': 10000,
            'x_gr': np.linspace(0, 10000, 21),
            'elevation_drop': 1000,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 200 * (1 - x_gr / 8000))
        
        forcing_params = {
            'T0': 15, 'P0': 2, 'gamma': 6.5e-3, 'mu': 0.65,
            'ts': 0, 'tf': 100
        }
        
        # Run models with different resolutions
        results = {}
        for name, config in [('base', config_base), ('fine', config_fine), ('coarse', config_coarse)]:
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
            forcing = TemperaturePrecipitationForcing(**forcing_params)
            model = flowline2d(config=config, geometry=geometry, forcing=forcing)
            results[name] = model.run()
        
        # Compare final lengths (should be within 0.1% of each other)
        base_length = results['base'].edge[-1]
        fine_length = results['fine'].edge[-1]
        coarse_length = results['coarse'].edge[-1]
        
        fine_error = abs(fine_length - base_length) / base_length
        coarse_error = abs(coarse_length - base_length) / base_length
        
        assert fine_error < 0.001, f"Fine grid error: {fine_error:.4f}"
        assert coarse_error < 0.001, f"Coarse grid error: {coarse_error:.4f}"


class TestBoundaryConditions:
    """Test boundary condition handling"""
    
    def test_glacier_head_boundary(self):
        """Test behavior at glacier head (upstream boundary)"""
        config = FlowlineConfig(delx=50, delt=0.0125/8, ts=0, tf=50, deltout=1)
        
        # Create geometry with very high mass balance at head
        basic_params = {
            'length': 5000,  # Shorter glacier for focused test
            'x_gr': np.linspace(0, 5000, 11),
            'elevation_drop': 500,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 4000))
        
        # High accumulation at head
        forcing = DirectMassBalanceForcing(b0=5, ts=0, tf=50)  # +5 m/yr everywhere
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Thickness at head should increase but remain finite
        head_thickness = result.h[:, 0]  # First grid point
        assert np.all(head_thickness >= 0), "Head thickness should be non-negative"
        assert np.all(np.isfinite(head_thickness)), "Head thickness should be finite"
    
    def test_glacier_terminus_boundary(self):
        """Test behavior at glacier terminus"""
        config = FlowlineConfig(delx=50, delt=0.0125/8, ts=0, tf=50, deltout=1)
        
        basic_params = {
            'length': 5000,
            'x_gr': np.linspace(0, 5000, 11),
            'elevation_drop': 500,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 4000))
        
        # Strong ablation to test terminus retreat
        forcing = DirectMassBalanceForcing(b0=-2, ts=0, tf=50)  # -2 m/yr everywhere
        
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
        config = FlowlineConfig(delx=50, delt=0.0125/4, ts=0, tf=20, deltout=1)
        
        basic_params = {
            'length': 5000,
            'x_gr': np.linspace(0, 5000, 11),
            'elevation_drop': 500,
            'width': 1000
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 4000))
        
        # Uniform mass balance
        forcing = DirectMassBalanceForcing(b0=1, ts=0, tf=20)  # +1 m/yr everywhere
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Calculate mass balance and volume change
        dt = np.diff(result.t)
        
        for i in range(1, len(result.t)):
            # Volume change
            edge_idx = result.edge_idx[i]
            if edge_idx > 0:
                vol_old = np.sum(result.h[i-1, :edge_idx] * result.w[:edge_idx] * config.delx)
                vol_new = np.sum(result.h[i, :edge_idx] * result.w[:edge_idx] * config.delx)
                dvol_dt = (vol_new - vol_old) / dt[i-1]
                
                # Mass balance input
                mb_input = result.gwb[i]
                
                # Should be approximately equal (within numerical precision)
                if abs(mb_input) > 1e-6:  # Avoid division by very small numbers
                    relative_error = abs(dvol_dt - mb_input) / abs(mb_input)
                    assert relative_error < 0.01, \
                        f"Mass conservation error at t={result.t[i]:.1f}: {relative_error:.4f}"


class TestOutputFormats:
    """Test output format conversions"""
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample model result for testing"""
        config = FlowlineConfig(delx=100, delt=0.0125/2, ts=0, tf=10, deltout=2)
        
        basic_params = {
            'length': 2000,
            'x_gr': np.linspace(0, 2000, 5),
            'elevation_drop': 200,
            'width': 500
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 50 * (1 - x_gr / 1500))
        
        forcing = DirectMassBalanceForcing(b0=0.5, ts=0, tf=10)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        return model.run()
    
    def test_to_pandas(self, sample_result):
        """Test conversion to pandas DataFrame"""
        df = sample_result.to_pandas()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_result.t)
        assert 'area' in df.columns
        assert 'bal' in df.columns
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
        assert 'b' in ds.data_vars
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
        config = FlowlineConfig(delx=50, delt=0.0125/4, ts=0, tf=20, deltout=5)
        
        basic_params = {
            'length': 5000,
            'x_gr': np.linspace(0, 5000, 11),
            'elevation_drop': 500,
            'width': 1000  # This will be overridden
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_variable_width(basic_params)
        h_init = np.maximum(0, 100 * (1 - x_gr / 4000))
        
        forcing = DirectMassBalanceForcing(b0=0, ts=0, tf=20)
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
        config = FlowlineConfig(delx=100, delt=0.0125/2, ts=0, tf=20, deltout=5)
        
        basic_params = {
            'length': 3000,
            'x_gr': np.linspace(0, 3000, 7),
            'elevation_drop': 300,
            'width': 800
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 80 * (1 - x_gr / 2500))
        
        # PDD forcing parameters
        forcing = TemperaturePrecipitationForcing(
            T0=15, P0=2, gamma=6.5e-3, mu=0.005,  # 5 mm/dd
            T2melt='pdd', pdd_Tamp=10, ts=0, tf=20
        )
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Should have PDD output
        assert hasattr(result, 'pdd'), "Result should contain PDD data"
        assert result.pdd is not None, "PDD data should not be None"
        assert np.all(result.pdd >= 0), "PDD values should be non-negative"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_geometry_raises_error(self):
        """Test that invalid geometry raises appropriate errors"""
        # Empty arrays should raise an error
        with pytest.raises((ValueError, IndexError)):
            geometry = FlowlineGeometry([], [], [])
            geometry.setup_grid(50)
    
    def test_mismatched_geometry_arrays(self):
        """Test that mismatched geometry arrays raise errors"""
        x_gr = np.linspace(0, 1000, 5)
        zb_gr = np.linspace(100, 0, 4)  # Wrong length
        w_geom = np.linspace(500, 500, 5)
        
        with pytest.raises((ValueError, IndexError)):
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
            geometry.setup_grid(50)
    
    def test_extreme_mass_balance_handling(self):
        """Test handling of extreme mass balance values"""
        config = FlowlineConfig(delx=100, delt=0.0125/8, ts=0, tf=5, deltout=1)
        
        basic_params = {
            'length': 2000,
            'x_gr': np.linspace(0, 2000, 5),
            'elevation_drop': 200,
            'width': 500
        }
        x_gr, zb_gr, w_geom = TestGeometry().create_uniform_slope(basic_params)
        h_init = np.maximum(0, 50 * (1 - x_gr / 1500))
        
        # Extremely negative mass balance
        forcing = DirectMassBalanceForcing(b0=-50, ts=0, tf=5)  # -50 m/yr
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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
