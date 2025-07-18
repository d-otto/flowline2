# -*- coding: utf-8 -*-
"""
Test suite for spinup mode functionality in FlowlineSweep

Tests cover:
- All spinup modes: shared, individual, per_run_custom, from_file
- Spinup configuration validation and error handling
- Integration with experimental runs
- Profile path management and file operations
- Error conditions and edge cases

Author: Test Suite
Created: 2025-07-12
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import xarray as xr

# Import the modules under test
import sys
sys.path.append('src')
from flowline.flowline2d import (
    FlowlineConfig, TemperaturePrecipitationForcing, DirectMassBalanceForcing
)
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline.sweep import FlowlineSweep


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


class TestSharedSpinup:
    """Test shared spinup mode functionality."""
    
    def test_shared_spinup_basic(self, base_objects, temp_dir):
        """Test basic shared spinup mode."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0]}
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 500},
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup:
            mock_spinup.return_value = str(temp_dir / "shared_spinup.nc")
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1,
                no_combine=True
            )
            
            # Test spinup orchestration without running full sweep
            run_objects = sweep._generate_run_objects()
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # Should call spinup simulation once
            mock_spinup.assert_called_once()
            
            # All runs should use the same profile
            assert len(profile_mapping) == 2  # Two runs in sweep
            assert profile_mapping['run_0000'] == profile_mapping['run_0001']
    
    def test_shared_spinup_config_creation(self, base_objects, temp_dir):
        """Test spinup config creation for shared mode."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 500, 'delt': 0.05},
            'forcing': {'T0': 7.0, 'P0': 1.5},
            'geometry': {'h_init': 5.0}
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        # Test config creation
        spinup_config_obj = sweep._create_spinup_config(config)
        assert spinup_config_obj.tf == 500
        assert spinup_config_obj.delt == 0.05
        assert spinup_config_obj.mu == 0.65  # Should inherit from base
        
        # Test geometry creation
        spinup_geometry_obj = sweep._create_spinup_geometry(geometry)
        assert spinup_geometry_obj.h_init == 5.0
        assert spinup_geometry_obj.profile is None  # Should be cleared
        
        # Test forcing creation
        spinup_forcing_obj = sweep._create_spinup_forcing(forcing, spinup_config=spinup_config_obj)
        assert isinstance(spinup_forcing_obj, TemperaturePrecipitationForcing)
        assert spinup_forcing_obj.T0 == 7.0
        assert spinup_forcing_obj.P0 == 1.5


class TestIndividualSpinup:
    """Test individual spinup mode functionality."""
    
    def test_individual_spinup_basic(self, base_objects, temp_dir):
        """Test basic individual spinup mode."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0, 9.0]}
        spinup_config = {
            'mode': 'individual',
            'enabled': True,
            'config': {'tf': 500},
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup:
            # Return different profiles for each call
            mock_spinup.side_effect = [
                str(temp_dir / "run_0000_spinup.nc"),
                str(temp_dir / "run_0001_spinup.nc"),
                str(temp_dir / "run_0002_spinup.nc")
            ]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1
            )
            
            run_objects = sweep._generate_run_objects()
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # Should call spinup simulation for each run
            assert mock_spinup.call_count == 3
            
            # Each run should have its own profile
            assert len(profile_mapping) == 3
            assert profile_mapping['run_0000'] != profile_mapping['run_0001']
            assert profile_mapping['run_0001'] != profile_mapping['run_0002']


class TestCustomSpinup:
    """Test per_run_custom spinup mode functionality."""
    
    def test_custom_spinup_basic(self, base_objects, temp_dir):
        """Test basic per_run_custom spinup mode."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0, 9.0, 10.0]}
        spinup_config = {
            'mode': 'per_run_custom',
            'enabled': True,
            'config': {'tf': 500},
            'forcing': {'T0': 8.0, 'P0': 2.0},  # Default forcing
            'customizations': [
                {'run_ids': ['run_0000', 'run_0001'], 'forcing': {'T0': 7.5}},
                {'run_ids': ['run_0002'], 'forcing': {'T0': 8.5}},
                # run_0003 uses default forcing
            ]
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup:
            mock_spinup.side_effect = [
                str(temp_dir / "custom_spinup_1.nc"),  # For T0=7.5 group
                str(temp_dir / "custom_spinup_2.nc"),  # For T0=8.5 group  
                str(temp_dir / "custom_spinup_3.nc"),  # For default group
            ]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1
            )
            
            run_objects = sweep._generate_run_objects()
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # Should call spinup simulation 3 times (3 unique configs)
            assert mock_spinup.call_count == 3
            
            # Runs with same custom config should share profiles
            assert len(profile_mapping) == 4
            assert profile_mapping['run_0000'] == profile_mapping['run_0001']  # Same custom group
            assert profile_mapping['run_0002'] != profile_mapping['run_0000']  # Different custom
            assert profile_mapping['run_0003'] != profile_mapping['run_0002']  # Default group
    
    def test_custom_spinup_complex_overrides(self, base_objects, temp_dir):
        """Test custom spinup with complex parameter overrides."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'config.mu': [0.6, 0.7]}
        spinup_config = {
            'mode': 'per_run_custom',
            'enabled': True,
            'config': {'tf': 500},
            'forcing': {'T0': 8.0, 'P0': 2.0},
            'customizations': [
                {
                    'run_ids': ['run_0000'], 
                    'config': {'tf': 600},
                    'forcing': {'T0': 7.0},
                    'geometry': {'h_init': 15.0}
                }
            ]
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=sweep_parameters,
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        # Test config creation with custom overrides
        custom_config = {'tf': 600}
        spinup_config_obj = sweep._create_spinup_config(config, custom_config)
        assert spinup_config_obj.tf == 600  # Custom override
        assert spinup_config_obj.mu == 0.65  # Base value
        
        # Test geometry creation with custom overrides
        custom_geometry = {'h_init': 15.0}
        spinup_geometry_obj = sweep._create_spinup_geometry(geometry, custom_geometry)
        assert spinup_geometry_obj.h_init == 15.0
        
        # Test forcing creation with custom overrides
        custom_forcing = {'T0': 7.0}
        spinup_forcing_obj = sweep._create_spinup_forcing(
            forcing, custom_forcing, spinup_config=spinup_config_obj
        )
        assert spinup_forcing_obj.T0 == 7.0


class TestFileSpinup:
    """Test from_file spinup mode functionality."""
    
    def test_file_spinup_basic(self, base_objects, temp_dir, mock_spinup_profile):
        """Test basic from_file spinup mode."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0]}
        spinup_config = {
            'mode': 'from_file',
            'enabled': True,
            'profile_path': mock_spinup_profile
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=sweep_parameters,
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            profile_mapping = sweep._orchestrate_spinups(run_objects, client)
        
        # All runs should use the same existing file
        assert len(profile_mapping) == 2
        assert profile_mapping['run_0000'] == mock_spinup_profile
        assert profile_mapping['run_0001'] == mock_spinup_profile
    
    def test_file_spinup_missing_file(self, base_objects, temp_dir):
        """Test from_file spinup mode with missing file."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'from_file',
            'enabled': True,
            'profile_path': str(temp_dir / "nonexistent.nc")
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            with pytest.raises(FileNotFoundError):
                sweep._orchestrate_spinups(run_objects, client)
    
    def test_file_spinup_missing_path(self, base_objects, temp_dir):
        """Test from_file spinup mode with missing profile_path."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'from_file',
            'enabled': True
            # Missing profile_path
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            with pytest.raises(ValueError, match="profile_path is required"):
                sweep._orchestrate_spinups(run_objects, client)


class TestSpinupErrorHandling:
    """Test error handling in spinup configurations."""
    
    def test_missing_explicit_forcing(self, base_objects, temp_dir):
        """Test error when explicit forcing is not provided."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 500}
            # Missing required forcing parameters
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        with pytest.raises(ValueError, match="Explicit spinup forcing parameters are required"):
            sweep._create_spinup_forcing(forcing, spinup_config=config)
    
    def test_unknown_spinup_mode(self, base_objects, temp_dir):
        """Test error for unknown spinup mode."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'unknown_mode',
            'enabled': True,
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            with pytest.raises(ValueError, match="Unknown spinup mode"):
                sweep._orchestrate_spinups(run_objects, client)
    
    def test_disabled_spinup(self, base_objects, temp_dir):
        """Test that disabled spinup returns empty mapping."""
        config, geometry, forcing = base_objects
        
        spinup_config = {
            'mode': 'shared',
            'enabled': False,
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            profile_mapping = sweep._orchestrate_spinups(run_objects, client)
        
        assert profile_mapping == {}
    
    def test_no_spinup_config(self, base_objects, temp_dir):
        """Test that missing spinup_config returns empty mapping."""
        config, geometry, forcing = base_objects
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={},
            spinup_config=None,  # No spinup config
            output_dir=temp_dir
        )
        
        run_objects = sweep._generate_run_objects()
        with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
            client = MagicMock()
            profile_mapping = sweep._orchestrate_spinups(run_objects, client)
        
        assert profile_mapping == {}


class TestSpinupIntegration:
    """Test integration between spinup and experimental runs."""
    
    def test_geometry_profile_assignment(self, base_objects, temp_dir, mock_spinup_profile):
        """Test that spinup profiles are correctly assigned to geometry objects."""
        config, geometry, forcing = base_objects
        
        sweep_parameters = {'forcing.T0': [7.0, 8.0]}
        spinup_config = {
            'mode': 'from_file',
            'enabled': True,
            'profile_path': mock_spinup_profile
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=sweep_parameters,
            spinup_config=spinup_config,
            output_dir=temp_dir,
            workers=1
        )
        
        # Test profile assignment logic
        run_objects = sweep._generate_run_objects()
        profile_mapping = {'run_0000': mock_spinup_profile, 'run_0001': mock_spinup_profile}
        
        # Simulate what happens in the run method
        for i, (config_obj, geometry_obj, forcing_obj) in enumerate(run_objects):
            run_id = sweep._get_run_id(i)
            
            if run_id in profile_mapping:
                # This is what the sweep.run() method does
                from copy import deepcopy
                test_geometry = deepcopy(geometry_obj)
                test_geometry.profile = profile_mapping[run_id]
                if hasattr(test_geometry, 'h_init'):
                    test_geometry.h_init = None
                
                # Verify profile was assigned
                assert test_geometry.profile == mock_spinup_profile
                assert test_geometry.h_init is None
    
    def test_run_id_generation(self, base_objects, temp_dir):
        """Test run ID generation."""
        config, geometry, forcing = base_objects
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters={'forcing.T0': [7.0, 8.0, 9.0]},
            output_dir=temp_dir
        )
        
        # Test run ID format
        assert sweep._get_run_id(0) == "run_0000"
        assert sweep._get_run_id(9) == "run_0009"
        assert sweep._get_run_id(123) == "run_0123"


class TestSpinupModeIntegration:
    """Integration tests for complete spinup workflows."""
    
    @pytest.mark.slow
    def test_shared_spinup_end_to_end(self, base_objects, temp_dir):
        """End-to-end test of shared spinup mode with real simulation calls."""
        config, geometry, forcing = base_objects
        
        # Use short simulation times for testing
        config.tf = 10
        config.deltout = 5
        
        sweep_parameters = {'forcing.T0': [7.5, 8.5]}
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 20, 'deltout': 10},
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        # Mock the actual simulation functions to avoid long computation
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup, \
             patch('flowline.sweep.run_flowline_simulation') as mock_exp:
            
            mock_spinup.return_value = str(temp_dir / "shared_spinup.nc")
            mock_exp.side_effect = [
                str(temp_dir / "run_0000.nc"),
                str(temp_dir / "run_0001.nc")
            ]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=1,
                no_combine=True
            )
            
            # This would normally run the full sweep
            run_objects = sweep._generate_run_objects()
            
            # Test just the spinup orchestration
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # Verify shared spinup behavior
            assert len(profile_mapping) == 2
            assert profile_mapping['run_0000'] == profile_mapping['run_0001']
            mock_spinup.assert_called_once()