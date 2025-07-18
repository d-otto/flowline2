"""
Simple Target Matching Example

This example shows the basic usage of the target matching system to optimize
glacier parameters to achieve a specific length.
"""

import numpy as np
from pathlib import Path

# Import flowline components
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
from flowline.spinup import FlowlineSpinup, LengthOnlyCost
import flowline.geometry as geometry_module


def main():
    """Simple target matching example."""
    
    print("Target Matching Example: Optimizing for 8km glacier length")
    print("=" * 60)
    
    # Create basic configuration (using stable parameters from basic_run)
    config = FlowlineConfig(
        ts=0,
        tf=1000,           # 500 years should be enough for spinup
        delt=0.0125 / 4, # Stable time step
        delx=50,          # 50m grid spacing
        deltout=1,        # Annual output
    )
    
    # Create simple geometry using available functions (same as basic_run)
    geom_params = {
        'domain_extent': 12000,
        'x_gr_points': 61,
        'elevation_drop': 1000,
        'width': 1000,
        'bed_characteristic_length': 10000,
    }
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(**geom_params)
    
    # Create initial ice thickness profile (simple wedge shape)
    h_init = np.maximum(0, 100 * (1 - x_gr / 5000))
    
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
    
    # Create forcing with initial guess (using stable parameters)
    forcing = TemperaturePrecipitationForcing(
        T0=8.5,         # Start with cooler temperature
        P0=2.0,
        gamma=6.5e-3,   # Lapse rate
        mu=0.65,        # Melt factor
        ts=config.ts,
        tf=config.tf,
    )
    
    # Configure target matching
    target_matching = {
        'targets': {
            'target_length': 8250,  # Want exactly 8km glacier
        },
        'adjustment_parameter': 'T0',        # Optimize temperature
        'cost_function': LengthOnlyCost,      # Simple length-only optimization
        'tolerance': 0.1,                     # Accept +/- delx error TODO: is this in units of length of temperature?
        'parameter_bounds': (7.0, 9.0),    # Extended temperature range
        'max_iterations': 100,                 # Limit optimization steps
        'max_simulation_time': config.tf           # Match the config.tf
    }
    
    # Create spinup with target matching
    spinup = FlowlineSpinup(
        config=config,
        geometry=geometry,
        forcing=forcing,
        target_matching=target_matching
    )
    
    # Run optimization
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initial T0: {forcing.T0:.3f}°C")
    print("Running optimization...")
    
    # Generate optimized profile
    profile_path = spinup.generate_profile(
        output_dir=output_dir,
        run_id="simple_target",
        no_progress=False
    )
    
    # Show results
    print(f"\nOptimization completed!")
    print(f"Final T0: {spinup.forcing.T0:.3f}°C")
    #print(f"Final length: {spinup.geometry.edge[-1]}")
    print(f"Profile saved to: {profile_path}")
    


if __name__ == "__main__":
    main()