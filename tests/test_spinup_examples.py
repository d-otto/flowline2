# -*- coding: utf-8 -*-
"""
Example tests demonstrating spinup mode usage patterns

These tests serve as both validation and documentation for how to use
the different spinup modes in FlowlineSweep.

Author: Test Suite  
Created: 2025-07-12
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_uniform_slope
from flowline.sweep import FlowlineSweep


@pytest.fixture
def example_objects():
    """Create realistic example objects for demonstrations."""
    # Configuration for a realistic but short-running example
    config = FlowlineConfig(
        ts=0, tf=50, delx=25, delt=0.05, deltout=5,
        mu=0.65, gamma=6.5e-3
    )
    
    # Create a realistic glacier geometry
    x_gr, zb_gr, w_geom = create_uniform_slope(
        domain_extent=3000, x_gr_points=121, elevation_drop=200,
        width=400, bed_characteristic_length=2000
    )
    h_init = np.maximum(0, 30.0 * (1 - x_gr / 2000))  # Initial ice profile
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=h_init)
    
    # Base forcing for equilibrium around current climate
    forcing = TemperaturePrecipitationForcing(
        ts=0, tf=50, T0=8.0, P0=2.0, gamma=6.5e-3, mu=0.65
    )
    
    return config, geometry, forcing


@pytest.fixture
def temp_dir():
    """Create temporary directory for example outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestSpinupExamples:
    """Example usage patterns for different spinup modes."""
    
    def test_example_shared_spinup_temperature_sensitivity(self, example_objects, temp_dir):
        """
        Example: Temperature sensitivity study with shared spinup.
        
        This demonstrates using a single steady-state profile as the starting
        point for multiple temperature scenarios. Perfect for climate sensitivity
        studies where all runs should start from the same initial state.
        """
        config, geometry, forcing = example_objects
        
        # Temperature sensitivity sweep (warming scenarios)
        sweep_parameters = {
            'forcing.T0': [8.0, 8.5, 9.0, 9.5, 10.0]  # Warming scenarios
        }
        
        # Shared spinup configuration
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 500, 'deltout': 50},  # Long spinup for steady state
            'forcing': {'T0': 8.0, 'P0': 2.0}      # Current climate conditions
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup, \
             patch('flowline.sweep.run_flowline_simulation') as mock_exp:
            
            mock_spinup.return_value = str(temp_dir / "equilibrium_profile.nc")
            mock_exp.side_effect = [str(temp_dir / f"warming_{i}.nc") for i in range(5)]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=4
            )
            
            # Verify the sweep configuration 
            run_objects = sweep._generate_run_objects()
            assert len(run_objects) == 5  # Five temperature scenarios
            
            # Test spinup orchestration
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # All runs should use the same equilibrium profile
            assert len(profile_mapping) == 5
            assert all(path == str(temp_dir / "equilibrium_profile.nc") 
                      for path in profile_mapping.values())
            
            # Only one spinup simulation should be run
            mock_spinup.assert_called_once()
    
    def test_example_individual_spinup_glacier_geometry_study(self, example_objects, temp_dir):
        """
        Example: Glacier geometry study with individual spinups.
        
        This demonstrates studying how different glacier geometries respond to
        the same climate. Each geometry needs its own spinup to reach equilibrium
        under current conditions before the experimental perturbation.
        """
        config, geometry, forcing = example_objects
        
        # Glacier width sensitivity (different geometries)
        sweep_parameters = {
            'geometry.w_scale': [0.8, 1.0, 1.2]  # Narrow, normal, wide glaciers
        }
        
        # Individual spinup configuration - each geometry needs its own equilibrium
        spinup_config = {
            'mode': 'individual',
            'enabled': True,
            'config': {'tf': 400, 'deltout': 40},
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup, \
             patch('flowline.sweep.run_flowline_simulation') as mock_exp:
            
            mock_spinup.side_effect = [
                str(temp_dir / "narrow_equilibrium.nc"),
                str(temp_dir / "normal_equilibrium.nc"), 
                str(temp_dir / "wide_equilibrium.nc")
            ]
            mock_exp.side_effect = [str(temp_dir / f"geometry_{i}.nc") for i in range(3)]
            
            # Mock the geometry modification for this example
            with patch.object(FlowlineGeometry, 'w_scale', create=True):
                sweep = FlowlineSweep(
                    base_config=config,
                    base_geometry=geometry,
                    base_forcing=forcing,
                    sweep_parameters=sweep_parameters,
                    spinup_config=spinup_config,
                    output_dir=temp_dir,
                    workers=2
                )
                
                run_objects = sweep._generate_run_objects()
                
                with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                    client = MagicMock()
                    profile_mapping = sweep._orchestrate_spinups(run_objects, client)
                
                # Each run should have its own unique equilibrium profile
                assert len(profile_mapping) == 3
                profiles = list(profile_mapping.values())
                assert len(set(profiles)) == 3  # All profiles are unique
                
                # Three spinup simulations should be run
                assert mock_spinup.call_count == 3
    
    def test_example_custom_spinup_target_length_study(self, example_objects, temp_dir):
        """
        Example: Target length study with custom spinups.
        
        This demonstrates a study where different glacier groups need different
        spinup conditions to achieve target lengths, then all experience the
        same climate scenario. Perfect for studying glaciers of similar size
        but different flow characteristics.
        """
        config, geometry, forcing = example_objects
        
        # Flow parameter study (same target length, different flow)
        sweep_parameters = {
            'config.fs': [1e-20, 5e-20, 1e-19]  # Different sliding parameters
        }
        
        # Custom spinup to achieve similar glacier lengths  
        spinup_config = {
            'mode': 'per_run_custom',
            'enabled': True,
            'config': {'tf': 600, 'deltout': 60},
            'forcing': {'T0': 8.0, 'P0': 2.0},  # Default forcing
            'customizations': [
                # Fast sliding glaciers need colder conditions to maintain length
                {'run_ids': ['run_0002'], 'forcing': {'T0': 7.5}},
                # Medium sliding uses default conditions  
                {'run_ids': ['run_0001'], 'forcing': {'T0': 8.0}},
                # Slow sliding needs warmer conditions to reach target length
                {'run_ids': ['run_0000'], 'forcing': {'T0': 8.5}}
            ]
        }
        
        with patch('flowline.sweep.run_spinup_simulation') as mock_spinup, \
             patch('flowline.sweep.run_flowline_simulation') as mock_exp:
            
            mock_spinup.side_effect = [
                str(temp_dir / "warm_spinup.nc"),    # For slow sliding (T0=8.5)
                str(temp_dir / "normal_spinup.nc"),  # For medium sliding (T0=8.0)  
                str(temp_dir / "cold_spinup.nc")     # For fast sliding (T0=7.5)
            ]
            mock_exp.side_effect = [str(temp_dir / f"flow_{i}.nc") for i in range(3)]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=3
            )
            
            run_objects = sweep._generate_run_objects()
            
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # Each run should have its own custom spinup profile
            assert len(profile_mapping) == 3
            # Verify that all runs have profiles assigned
            profiles = list(profile_mapping.values())
            assert len(set(profiles)) == 3  # All profiles are unique
            assert all(str(temp_dir) in path for path in profiles)
            
            # Three unique spinup configurations should be run
            assert mock_spinup.call_count == 3
    
    def test_example_file_spinup_benchmark_study(self, example_objects, temp_dir):
        """
        Example: Benchmark study using pre-computed spinup.
        
        This demonstrates using a previously computed steady-state profile
        as the starting point for all runs. Useful for reproducible studies
        or when you have a high-quality reference state.
        """
        config, geometry, forcing = example_objects
        
        # Create a mock reference profile
        reference_profile = temp_dir / "reference_steady_state.nc"
        # In practice, this would be a real NetCDF file with glacier profile
        reference_profile.touch()
        
        # Model intercomparison study (different parameter combinations)
        sweep_parameters = {
            'config.mu': [0.6, 0.65, 0.7],
            'config.gamma': [6.0e-3, 6.5e-3, 7.0e-3]
        }
        
        # Use existing reference profile for all runs
        spinup_config = {
            'mode': 'from_file',
            'enabled': True,
            'profile_path': str(reference_profile)
        }
        
        with patch('flowline.sweep.run_flowline_simulation') as mock_exp:
            mock_exp.side_effect = [str(temp_dir / f"benchmark_{i}.nc") for i in range(9)]
            
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=sweep_parameters,
                spinup_config=spinup_config,
                output_dir=temp_dir,
                workers=4
            )
            
            run_objects = sweep._generate_run_objects()
            assert len(run_objects) == 9  # 3 x 3 parameter combinations
            
            with patch('dask.distributed.Client'), patch('dask.distributed.LocalCluster'):
                client = MagicMock()
                profile_mapping = sweep._orchestrate_spinups(run_objects, client)
            
            # All runs should use the same reference profile
            assert len(profile_mapping) == 9
            assert all(path == str(reference_profile) for path in profile_mapping.values())
            
            # No spinup simulations should be run (using existing file)
            with patch('flowline.sweep.run_spinup_simulation') as mock_spinup:
                # Re-run orchestration to verify no spinup calls
                sweep._orchestrate_spinups(run_objects, client)
                mock_spinup.assert_not_called()


class TestSpinupWorkflowPatterns:
    """Test common workflow patterns and best practices."""
    
    def test_workflow_climate_impact_assessment(self, example_objects, temp_dir):
        """
        Workflow example: Climate impact assessment.
        
        Pattern: Shared equilibrium + multiple climate scenarios
        Use case: How does glacier respond to different warming scenarios?
        """
        config, geometry, forcing = example_objects
        
        # Multiple climate scenarios from current to +3°C warming
        climate_scenarios = {
            'forcing.T0': [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]
        }
        
        spinup_config = {
            'mode': 'shared',
            'enabled': True,
            'config': {'tf': 1000, 'deltout': 100},  # Long equilibrium run
            'forcing': {'T0': 8.0, 'P0': 2.0}        # Current climate baseline
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=climate_scenarios,
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        # This pattern is efficient because:
        # 1. Single long spinup run establishes current equilibrium
        # 2. All climate scenarios start from same baseline
        # 3. Direct comparison of climate sensitivity
        assert sweep.spinup_config['mode'] == 'shared'
        assert len(sweep._generate_run_objects()) == 7
    
    def test_workflow_glacier_geometry_intercomparison(self, example_objects, temp_dir):
        """
        Workflow example: Glacier geometry intercomparison.
        
        Pattern: Individual spinups + geometry variations
        Use case: How do different bed topographies affect glacier response?
        """
        config, geometry, forcing = example_objects
        
        # Different bed slope configurations
        geometry_variations = {
            'geometry.bed_slope': [0.03, 0.05, 0.07, 0.10]
        }
        
        spinup_config = {
            'mode': 'individual',
            'enabled': True,
            'config': {'tf': 800, 'deltout': 80},
            'forcing': {'T0': 8.0, 'P0': 2.0}
        }
        
        # Mock the bed_slope attribute for this example
        with patch.object(FlowlineGeometry, 'bed_slope', create=True):
            sweep = FlowlineSweep(
                base_config=config,
                base_geometry=geometry,
                base_forcing=forcing,
                sweep_parameters=geometry_variations,
                spinup_config=spinup_config,
                output_dir=temp_dir
            )
            
            # This pattern is necessary because:
            # 1. Each geometry reaches different equilibrium size
            # 2. Individual spinups normalize for geometry differences
            # 3. Experimental response isolates geometric effects
            assert sweep.spinup_config['mode'] == 'individual'
            assert len(sweep._generate_run_objects()) == 4
    
    def test_workflow_model_parameter_optimization(self, example_objects, temp_dir):
        """
        Workflow example: Model parameter optimization.
        
        Pattern: Custom spinups for target matching + parameter sweep
        Use case: Find parameter combinations that match observed glacier length
        """
        config, geometry, forcing = example_objects
        
        # Parameter combinations to test
        parameter_sweep = {
            'config.fs': [1e-20, 5e-20, 1e-19],
            'config.mu': [0.6, 0.65, 0.7]
        }
        
        spinup_config = {
            'mode': 'per_run_custom', 
            'enabled': True,
            'config': {'tf': 1200, 'deltout': 120},
            'forcing': {'T0': 8.0, 'P0': 2.0},
            'customizations': [
                # Adjust spinup temperature to achieve target length for each parameter set
                {'run_ids': ['run_0000', 'run_0003', 'run_0006'], 'forcing': {'T0': 7.8}},  # Low fs needs cooler
                {'run_ids': ['run_0001', 'run_0004', 'run_0007'], 'forcing': {'T0': 8.0}},  # Medium fs uses default
                {'run_ids': ['run_0002', 'run_0005', 'run_0008'], 'forcing': {'T0': 8.2}}   # High fs needs warmer
            ]
        }
        
        sweep = FlowlineSweep(
            base_config=config,
            base_geometry=geometry,
            base_forcing=forcing,
            sweep_parameters=parameter_sweep,
            spinup_config=spinup_config,
            output_dir=temp_dir
        )
        
        # This pattern enables:
        # 1. All glaciers start at same target length despite parameter differences
        # 2. Clean comparison of parameter effects on dynamics
        # 3. Automated optimization workflows 
        assert sweep.spinup_config['mode'] == 'per_run_custom'
        assert len(sweep._generate_run_objects()) == 9  # 3 x 3 parameter combinations
        assert len(spinup_config['customizations']) == 3  # Three fs groups