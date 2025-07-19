"""
Target Matching Test

Simple test to verify that target matching works correctly with multiple
FlowlineSpinup objects that should optimize to different parameters.

This test creates two glaciers with different melt factors (mu) and uses
target matching to optimize temperature so both achieve the same target length.
"""

import numpy as np
from pathlib import Path
import sys
import logging

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep
from src.flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from src.flowline.geometry import FlowlineGeometry
from src.flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector
import src.flowline.geometry as geometry_module

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """Test target matching with two different melt factors."""
    
    print("Target Matching Test: Two melt factors targeting same length")
    print("=" * 60)
    
    # Use exact settings from target_matching_simple example (for reference)
    
    # Create simple geometry (same as target_matching_simple)
    geom_params = {
        'domain_extent': 12000,
        'x_gr_points': 61,
        'elevation_drop': 1000,
        'width': 1000,
        'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(**geom_params)
    h_init = np.maximum(0, 100 * (1 - x_gr / 5000))  # Simple wedge shape
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
    
    # Target length
    target_length = 8000  # Same as target_matching_simple
    
    # Create two different melt factors (use smaller values that are more realistic)
    gamma_values = [0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]  # Different melt factors
    
    # Create spinup objects for each melt factor
    spinup_objects = {}
    experimental_perturbations = {}
    
    for i, gamma_val in enumerate(gamma_values):
        run_id = f"run_{i:04d}"
        print(f"Creating spinup for {run_id} with mu={gamma_val}")
        
        # Create independent config for this run
        spinup_config = FlowlineConfig(
            ts=0,
            tf=1000,
            delt=0.0125 / 4,
            delx=50,
            deltout=1,
        )
        
        # Create independent forcing with this melt factor
        spinup_forcing = TemperaturePrecipitationForcing(
            T0=8.,           # Initial guess temperature
            P0=2.0,
            mu=0.65,     # Lapse rate
            gamma=gamma_val,        # Different melt factor for each run
            ts=spinup_config.ts,
            tf=spinup_config.tf,
        )
        
        # Configure target matching (same as target_matching_simple)
        target_matching = {
            'targets': {
                'target_length': target_length,
            },
            'adjustment_parameter': 'T0',        # Optimize temperature
            'cost_function': LengthOnlyCost,
            'steady_state_detector': VolumeChangeRateDetector,
            'tolerance': 0.05,                    # 0.05°C temperature tolerance for better precision
            'parameter_bounds': (5.0, 9.0),    # Wider temperature range to avoid bound constraints
            'max_iterations': 150,               # More iterations for better convergence
            'max_simulation_time': spinup_config.tf
        }
        
        # Create FlowlineSpinup object
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching=target_matching
        )
        
        spinup_objects[run_id] = spinup_obj
        logger.debug(f"Created spinup {run_id} with initial T0={spinup_obj.forcing.T0:.3f}, gamma={spinup_obj.forcing.gamma:.6f}")
        
        # Small temperature perturbation for experimental run
        experimental_perturbations[run_id] = {
            'forcing.T0': lambda T0: T0 + 0.5  # +0.5°C warming
        }
    
    # Create experimental config and forcing (will be modified by perturbations)
    experimental_config = FlowlineConfig(
        ts=0,
        tf=500,           # Shorter experimental run
        delt=0.0125 / 4,
        delx=50,
        deltout=1,
    )
    
    experimental_forcing = TemperaturePrecipitationForcing(
        T0=8.0,
        P0=2.0,
        mu=0.65,          
        gamma=0.0065,
        ts=experimental_config.ts,
        tf=experimental_config.tf,
    )
    
    # Set up output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("\\nRunning target matching test...")
    print(f"Expected: Both runs should achieve ~{target_length}m length with different T0 values")
    
    # Run sweep with target matching
    sweep = FlowlineSweep(
        base_config=experimental_config,
        base_geometry=geometry,
        base_forcing=experimental_forcing,
        sweep_parameters={},  # No additional sweep parameters
        spinup_objects=spinup_objects,
        experimental_perturbations=experimental_perturbations,
        output_dir=str(output_dir),
        workers=8,
        no_progress=True  # Disable progress bars for cleaner output
    )
    
    sweep.run()
    
    # Check results
    print("\\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Load and check initial lengths
    results_file = output_dir / "combined_results.nc"
    if results_file.exists():
        import xarray as xr
        ds = xr.open_dataset(results_file)
        
        for i, run_id in enumerate(ds.run_id.values):
            initial_length = ds.edge.sel(run_id=run_id).isel(time=0).values
            gamma_val = gamma_values[i]
            print(f"Run {run_id} (mu={gamma_val}): Initial length = {initial_length:.1f}m")
        
        ds.close()
        
        # Success criteria
        lengths = []
        for i, run_id in enumerate(ds.run_id.values):
            lengths.append(ds.edge.sel(run_id=run_id).isel(time=0).values)
        
        if all(abs(length - target_length) < 50 for length in lengths):
            print("\\n✓ SUCCESS: Both runs achieved target length within 50m")
        else:
            print("\\n✗ FAILURE: Runs did not achieve consistent target length")
            print(f"Target: {target_length}m")
            print(f"Actual: {[f'{length:.1f}m' for length in lengths]}")
    else:
        print("✗ FAILURE: No results file found")
    
    print("="*60)


if __name__ == "__main__":
    main()