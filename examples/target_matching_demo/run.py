"""
Target Matching Demonstration

This example shows how to use the new target matching optimization system
in FlowlineSpinup to automatically find parameter values that achieve 
specific glacier targets (e.g., length, average thickness).
"""

import numpy as np
from pathlib import Path

# Import flowline components
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
from flowline.spinup import FlowlineSpinup, LengthOnlyCost, LengthAndAverageThicknessCost, VolumeChangeRateDetector
import flowline.geometry as geometry_module

# Import CLI utilities
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs


def main():
    """Main function demonstrating target matching."""
    
    # Parse CLI arguments
    args = parse_sweep_cli_args("Target matching optimization demonstration")
    
    # =============================================================================
    # SETUP BASE CONFIGURATION
    # =============================================================================
    
    # Create base configuration
    base_config = FlowlineConfig(
        ts=0,
        tf=1000,  # Long spinup time for optimization
        delx=25,
        deltout=1,
        min_thick=1.0
    )
    
    # Create geometry (using existing geometry function)
    geom_params = {
        'domain_extent': 15000,
        'x_gr_points': 75,
        'elevation_drop': 500,
        'width': 1000,
        'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(**geom_params)
    
    # Create initial ice thickness profile
    h_init = np.maximum(0, 100 * (1 - x_gr / 5000))
    
    base_geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
    
    # Create base forcing
    base_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=1000,
        P0=2000,    # 2000 mm/year precipitation
        T0=8.0,     # 8°C temperature (will be optimized)
        gamma=6.5,  # 6.5°C/km lapse rate
        mu=3.0      # 3.0 mm/°C/day melt factor
    )
    
    # =============================================================================
    # EXAMPLE 1: LENGTH-ONLY TARGET MATCHING
    # =============================================================================
    
    print("=" * 60)
    print("EXAMPLE 1: LENGTH-ONLY TARGET MATCHING")
    print("=" * 60)
    
    # Configure target matching for length optimization
    length_target_matching = {
        'targets': {
            'target_length': 8000,  # Target: 8km glacier
        },
        'adjustment_parameter': 'T0',                    # Optimize temperature
        'cost_function': LengthOnlyCost,                  # Only consider length
        'steady_state_detector': VolumeChangeRateDetector,   # Use dV/dt detector
        'tolerance': 100,                                # Accept ±100m error
        'parameter_bounds': (5.0, 12.0),                # T0 search bounds
        'max_iterations': 10,                            # Max optimization steps
        'max_simulation_time': 1000                      # Max simulation time
    }
    
    # Create spinup with target matching
    length_spinup = FlowlineSpinup(
        config=base_config,
        geometry=base_geometry,
        forcing=base_forcing,
        target_matching=length_target_matching
    )
    
    # Generate optimized profile
    output_dir = Path(args.output_dir) / "length_only_target"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Optimizing T0 to achieve 8km glacier length...")
    profile_path = length_spinup.generate_profile(
        output_dir=output_dir,
        run_id="length_target_demo",
        no_progress=args.no_progress
    )
    
    print(f"Length-only optimization completed!")
    print(f"Profile saved to: {profile_path}")
    print(f"Final T0 value: {length_spinup.forcing.T0:.3f}°C")
    
    # =============================================================================
    # EXAMPLE 2: LENGTH AND THICKNESS TARGET MATCHING
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: LENGTH AND THICKNESS TARGET MATCHING")
    print("=" * 60)
    
    # Configure target matching for length + thickness optimization
    length_thickness_target_matching = {
        'targets': {
            'target_length': 8000,         # Target: 8km glacier
            'target_avg_thickness': 120,   # Target: 120m average thickness
        },
        'adjustment_parameter': 'T0',
        'cost_function': LengthAndAverageThicknessCost(length_weight=1.0, thickness_weight=1.0),
        'steady_state_detector': VolumeChangeRateDetector(threshold=5e-7, window_size=30, min_time=200),
        'tolerance': 50,                   # Tighter tolerance
        'parameter_bounds': (5.0, 12.0),
        'max_iterations': 15,
        'max_simulation_time': 1500
    }
    
    # Create fresh forcing object for second example
    forcing_copy = TemperaturePrecipitationForcing(
        ts=0,
        tf=1500,
        P0=2000,
        T0=8.0,     # Reset to original value
        gamma=6.5,
        mu=3.0
    )
    
    # Create spinup with multi-target matching
    length_thickness_spinup = FlowlineSpinup(
        config=FlowlineConfig(
            ts=0,
            tf=1500,
            delx=25,
            deltout=1,
            min_thick=1.0
        ),
        geometry=base_geometry,
        forcing=forcing_copy,
        target_matching=length_thickness_target_matching
    )
    
    # Generate optimized profile
    output_dir = Path(args.output_dir) / "length_thickness_target"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Optimizing T0 to achieve 8km length AND 120m average thickness...")
    profile_path = length_thickness_spinup.generate_profile(
        output_dir=output_dir,
        run_id="length_thickness_demo",
        no_progress=args.no_progress
    )
    
    print(f"Length + thickness optimization completed!")
    print(f"Profile saved to: {profile_path}")
    print(f"Final T0 value: {length_thickness_spinup.forcing.T0:.3f}°C")
    
    # =============================================================================
    # EXAMPLE 3: CUSTOM COST FUNCTION
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CUSTOM COST FUNCTION")
    print("=" * 60)
    
    # Define custom cost function
    def custom_volume_cost(model_state, targets):
        """Custom cost function that targets glacier volume."""
        # Calculate current volume
        h = model_state['h']
        delx = model_state['delx']
        
        # Simple volume approximation
        current_volume = np.sum(h[h > 0]) * delx * 1000  # Assume 1km width
        target_volume = targets['target_volume']
        
        # Return absolute difference
        return abs(current_volume - target_volume)
    
    # Configure custom target matching
    custom_target_matching = {
        'targets': {
            'target_volume': 5e8,  # Target: 500 million m³
        },
        'adjustment_parameter': 'T0',
        'cost_function': custom_volume_cost,  # Use custom function
        'steady_state_detector': VolumeChangeRateDetector,
        'tolerance': 1e7,  # Accept ±10 million m³ error
        'parameter_bounds': (5.0, 12.0),
        'max_iterations': 12,
        'max_simulation_time': 1200
    }
    
    # Create fresh forcing object for third example
    forcing_copy2 = TemperaturePrecipitationForcing(
        ts=0,
        tf=1200,
        P0=2000,
        T0=8.0,
        gamma=6.5,
        mu=3.0
    )
    
    # Create spinup with custom cost function
    custom_spinup = FlowlineSpinup(
        config=FlowlineConfig(
            ts=0,
            tf=1200,
            delx=25,
            deltout=1,
            min_thick=1.0
        ),
        geometry=base_geometry,
        forcing=forcing_copy2,
        target_matching=custom_target_matching
    )
    
    # Generate optimized profile
    output_dir = Path(args.output_dir) / "custom_target"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Optimizing T0 to achieve target volume using custom cost function...")
    profile_path = custom_spinup.generate_profile(
        output_dir=output_dir,
        run_id="custom_cost_demo",
        no_progress=args.no_progress
    )
    
    print(f"Custom cost function optimization completed!")
    print(f"Profile saved to: {profile_path}")
    print(f"Final T0 value: {custom_spinup.forcing.T0:.3f}°C")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Length-only target (8km):           T0 = {length_spinup.forcing.T0:.3f}°C")
    print(f"Length + thickness target:          T0 = {length_thickness_spinup.forcing.T0:.3f}°C")
    print(f"Custom volume target:               T0 = {custom_spinup.forcing.T0:.3f}°C")
    print("\nAll optimizations completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()