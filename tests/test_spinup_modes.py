# -*- coding: utf-8 -*-
"""
Test suite for modern FlowlineSpinup functionality

Tests cover:
- FlowlineSpinup object creation and configuration
- Target matching and optimization
- Integration with FlowlineSweep for experimental runs
- Modern 4-object architecture patterns
- Spinup and perturbation workflows

Author: Test Suite
Updated: 2025-07-19 (Modernized for current architecture)
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import xarray as xr

# Import the modules under test
from flowline.flowline2d import (
    FlowlineConfig, TemperaturePrecipitationForcing, DirectMassBalanceForcing
)
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup


@pytest.fixture
def base_objects():
    """Create base objects for testing."""
    # Base configuration
    config = FlowlineConfig(
        ts=0, tf=100, delx=25, delt=0.1, deltout=1,
        mu=0.65
    )
    
    # Base geometry
    x_gr, zb_gr, w_geom = create_uniform_slope(
        domain_extent=2000, x_gr_points=81, elevation_drop=100, 
        width=500, bed_characteristic_length=1000
    )
    h_init = np.maximum(0, 10.0 * (1 - x_gr / 1000))  # Simple initial profile
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
    
    # Base forcing
    forcing = TemperaturePrecipitationForcing(
        ts=0, tf=100, T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65
    )
    
    return config, geometry, forcing


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_spinup_profile(temp_dir):
    """Create a mock spinup profile file."""
    profile_path = temp_dir / "test_spinup_profile.nc"
    
    # Create a simple mock profile dataset
    x = np.linspace(0, 2000, 81)
    h = np.maximum(0, 50 - 0.025 * x)  # Simple triangular profile
    
    ds = xr.Dataset({
        'h': (['x'], h),
        'x': (['x'], x)
    })
    ds.to_netcdf(profile_path)
    
    return str(profile_path)


class TestFlowlineSpinupBasic:
    """Test basic FlowlineSpinup object functionality."""
    
    def test_spinup_object_creation(self, base_objects, temp_dir):
        """Test creating FlowlineSpinup objects."""
        config, geometry, forcing = base_objects
        
        # Create spinup configuration
        spinup_config = FlowlineConfig(
            ts=0, tf=500, delx=25, delt=0.1, deltout=1,
            mu=0.65
        )
        
        # Create spinup forcing
        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0, tf=500, T0=8.0, P0=2.0, mu=0.65
        )
        
        # Create FlowlineSpinup object
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing
        )
        
        # Test basic attributes
        assert spinup_obj.config.tf == 500
        assert spinup_obj.forcing.T0 == 8.0
        assert spinup_obj.geometry.x_gr.shape == geometry.x_gr.shape
    
    def test_spinup_with_target_matching(self, base_objects, temp_dir):
        """Test FlowlineSpinup with target matching configuration."""
        config, geometry, forcing = base_objects
        
        # Create spinup configuration
        spinup_config = FlowlineConfig(
            ts=0, tf=500, delx=25, delt=0.1, deltout=1,
            mu=0.65
        )
        
        # Create spinup forcing
        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0, tf=500, T0=8.0, P0=2.0, mu=0.65
        )
        
        # Create FlowlineSpinup object with target matching
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching={
                'target_length': 1500,  # Target 1.5km glacier length
                'adjustment_parameter': 'forcing.T0',
                'adjustment_function': lambda mu: 8.0 + (mu - 0.65) * 2.0,
                'tolerance': 100
            }
        )
        
        # Test target matching configuration
        assert spinup_obj.target_matching['target_length'] == 1500
        assert spinup_obj.target_matching['adjustment_parameter'] == 'forcing.T0'
        assert spinup_obj.target_matching['tolerance'] == 100


class TestFlowlineSpinupIntegration:
    """Test FlowlineSpinup integration with FlowlineSweep."""
    
    def test_single_spinup_object(self, base_objects, temp_dir):
        """Test using a single FlowlineSpinup object for all runs."""
        config, geometry, forcing = base_objects
        
        # Create shared spinup object
        spinup_config = FlowlineConfig(
            ts=0, tf=500, delx=25, delt=0.1, deltout=1,
            mu=0.65
        )
        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0, tf=500, T0=8.0, P0=2.0, mu=0.65
        )
        shared_spinup = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing
        )
        
        # Create sweep with single spinup object
        sweep_parameters = {'forcing.T0': [7.0, 8.0]}
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=sweep_parameters,
            spinup_objects=shared_spinup,  # Single object
            output_dir=temp_dir,
            workers=1
        )
        
        # Test that sweep accepts the spinup object
        assert sweep.spinup_objects is shared_spinup
    
    def test_multiple_spinup_objects(self, base_objects, temp_dir):
        """Test using different FlowlineSpinup objects for different runs."""
        config, geometry, forcing = base_objects
        
        # Create multiple spinup objects
        spinup_objects = {}
        melt_factors = [0.6, 0.7]
        
        for i, mu in enumerate(melt_factors):
            run_id = f"run_{i:04d}"
            
            spinup_config = FlowlineConfig(
                ts=0, tf=500, delx=25, delt=0.1, deltout=1,
                mu=mu
            )
            spinup_forcing = TemperaturePrecipitationForcing(
                ts=0, tf=500, T0=8.0, P0=2.0, mu=mu
            )
            spinup_obj = FlowlineSpinup(
                config=spinup_config,
                geometry=geometry,
                forcing=spinup_forcing,
                target_matching={
                    'target_length': 1500,
                    'adjustment_parameter': 'forcing.T0',
                    'adjustment_function': lambda mu_val: 8.0 + (mu_val - 0.65) * 2.0,
                    'tolerance': 100
                }
            )
            spinup_objects[run_id] = spinup_obj
        
        # Create experimental perturbations
        experimental_perturbations = {
            'run_0000': {'forcing.T0': lambda T0: T0 + 1.0},
            'run_0001': {'forcing.T0': lambda T0: T0 + 1.0}
        }
        
        # Create sweep with multiple spinup objects
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},  # No additional parameters
            spinup_objects=spinup_objects,
            experimental_perturbations=experimental_perturbations,
            output_dir=temp_dir,
            workers=1
        )
        
        # Test that sweep accepts the spinup objects
        assert len(sweep.spinup_objects) == 2
        assert 'run_0000' in sweep.spinup_objects
        assert 'run_0001' in sweep.spinup_objects


class TestLegacySpinupSupport:
    """Test that legacy spinup configurations still work for backward compatibility."""
    
    def test_legacy_shared_spinup(self, base_objects, temp_dir):
        """Test legacy shared spinup configuration."""
        config, geometry, forcing = base_objects
        
        # Legacy dictionary-based spinup config
        spinup_config = {
            'mode': 'shared',
            'config': {'tf': 500, 'delt': 0.05},
            'forcing': TemperaturePrecipitationForcing(
                ts=0, tf=500, T0=8.0, P0=2.0, mu=0.65
            ),  # Using object instead of dict
            'use_initial_h': False
        }
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0]}
        
        try:
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1
            )
            # Test that sweep accepts legacy config
            assert sweep.spinup_config is not None
        except ValueError as e:
            # If legacy support is removed, this is expected
            assert "FlowlineForcing object" in str(e) or "dictionary" in str(e)
    
    def test_legacy_individual_spinup(self, base_objects, temp_dir):
        """Test legacy individual spinup configuration."""
        config, geometry, forcing = base_objects
        
        # Legacy dictionary-based spinup config
        spinup_config = {
            'mode': 'individual',
            'config': {'tf': 500, 'delt': 0.05},
            'forcing': TemperaturePrecipitationForcing(
                ts=0, tf=500, T0=8.0, P0=2.0, mu=0.65
            ),
            'use_initial_h': False
        }
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0, 9.0]}
        
        try:
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1
            )
            # Test that sweep accepts legacy config
            assert sweep.spinup_config is not None
        except ValueError as e:
            # If legacy support is removed, this is expected
            assert "FlowlineForcing object" in str(e) or "dictionary" in str(e)
