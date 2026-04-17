#!/usr/bin/env python3
"""
Width profile variations example with target matching.

This example demonstrates the effect of different glacier width profiles on glacier
dynamics by comparing "top-heavy", "bottom-heavy", and "neutral" width profiles.
All glaciers are adjusted to achieve the same initial length (8000m) using target
matching to optimize T0, allowing for direct comparison of width profile effects.

Width profiles:
- Top-heavy: Wide at head (2000m), narrow at terminus (500m)
- Bottom-heavy: Narrow at head (500m), wide at terminus (2000m)
- Neutral: Constant width (1250m) - average of the extremes

Hourglass and oval shapes are compared in the separate width_shape_variations example.

The example uses FlowlineSpinup objects with target matching to ensure comparable
initial states, then tests glacier response to climate perturbations.
"""

from pathlib import Path
import numpy as np

from flowline.sweep import FlowlineSweep
from flowline.spinup import (
    FlowlineSpinup,
    VolumeOnlyCost,
    VolumeChangeRateDetector,
    LengthOnlyCost,
)
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module


def main():
    # Parse command line arguments
    args = parse_sweep_cli_args(
        "Width profile variations example with target matching."
    )

    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    # --- Base Configuration for Response Testing ---
    response_config = FlowlineConfig(
        ts=0,
        tf=100,  # Response test duration: 100 years (reduced for testing)
        delx=25,
        delt=0.00078125,
        deltout=1.0,
        min_thick=1.0,
    )

    # --- Define Common Geometry Parameters ---
    common_geom_params = {
        "domain_extent": 12000,
        "x_gr_points": 61,
        "elevation_drop": 1000,
        "bed_characteristic_length": 10000,
    }

    # --- Create Width Profile Geometries ---
    # All profiles satisfy these conditions over [0, 8km]:
    #   - top w(0) = bottom w(8km) = hourglass w(0) = hourglass w(8km) = 1750
    #   - bottom w(0) = top w(8km) = 750
    #   - top w(4km) = bottom w(4km) = hourglass w(4km) = neutral = 1250
    #   - equal cross-sectional area integrals (all = 10,000,000 m²)
    # w_term values follow from the linear formula:
    #   w(x) = w_head - (w_head - w_term) * (x / bed_characteristic_length)
    #   => w_term = w_head - (w_head - w(8km)) / 0.8

    # Top-heavy: Wide at head (1750m), narrow at 8km (750m)
    x_gr_top, zb_gr_top, w_geom_top = geometry_module.create_variable_width(
        w_head=1750, w_term=500, **common_geom_params
    )

    # Bottom-heavy: Narrow at head (750m), wide at 8km (1750m)
    x_gr_bottom, zb_gr_bottom, w_geom_bottom = geometry_module.create_variable_width(
        w_head=750, w_term=2000, **common_geom_params
    )

    # Neutral: Constant width (midpoint of top and bottom)
    x_gr_neutral, zb_gr_neutral, w_geom_neutral = geometry_module.create_uniform_slope(
        width=1250,
        **common_geom_params,
    )

    # Store geometries for analysis
    width_profiles = {
        "top_heavy": {
            "geometry_data": (x_gr_top, zb_gr_top, w_geom_top),
            "description": "Wide at head (1750m), narrow at 8km (750m)",
        },
        "bottom_heavy": {
            "geometry_data": (x_gr_bottom, zb_gr_bottom, w_geom_bottom),
            "description": "Narrow at head (750m), wide at 8km (1750m)",
        },
        "neutral": {
            "geometry_data": (x_gr_neutral, zb_gr_neutral, w_geom_neutral),
            "description": "Constant width (1250m)",
        },
    }

    # Create reasonable initial ice thickness profile for spinup
    scale = 100
    length = 5000
    h_init = np.maximum(
        0, scale * (1 - x_gr_top / length)
    )  # Same x_gr for all profiles

    # --- Base Forcing for Response Testing ---
    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,
        T0=7.0,  # Will be overridden by spinup optimization
        mu=0.6,
    )

    # --- Create FlowlineSpinup Objects for Each Width Profile ---
    spinup_objects = {}

    for profile_type, profile_info in width_profiles.items():
        x_gr, zb_gr, w_geom = profile_info["geometry_data"]

        # Create geometry object for this width profile
        geometry = FlowlineGeometry(
            x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, x_init=x_gr, h_init=h_init
        )

        # Spinup configuration
        spinup_config = FlowlineConfig(
            ts=0,
            tf=500,  # 500-year spinup (reduced for faster testing)
            delx=25,
            delt=0.00078125,
            deltout=1.0,
            min_thick=1.0,
        )

        # Spinup forcing
        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0,
            tf=1000,
            P0=2.0,
            T0=7.0,  # Will be adjusted by target matching
            mu=0.6,
        )

        # Create FlowlineSpinup with target matching to achieve same glacier volume.
        # Target volume is taken from the neutral profile initial state of the
        # previous area-matched simulations: 9.635e8 m³ (~0.9635 km³).
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching={
                "targets": {
                    # "target_volume": 9.635e8,  # m³ (~0.9635 km³)
                    "target_length": 8000,
                },
                "adjustment_parameters": ["T0"],  # Optimize temperature
                "bounds": [(5.5, 8.5)],  # Temperature bounds
                "cost_function": LengthOnlyCost,
                "steady_state_detector": VolumeChangeRateDetector,
                # "tolerance": 1e7,  # Accept ±1×10^7 m³ (~0.01 km³) from target
                "tolerance": 50,
                "max_simulation_time": 1000,
                "optimization_options": {
                    "maxfev": 25,  # Reduced function evaluations
                    "maxiter": 10,
                },
            },
        )

        spinup_objects[profile_type] = spinup_obj

    # --- Create Experimental Perturbations ---
    # Apply +1.5°C warming to test response sensitivity across width profiles
    experimental_perturbations = {}
    for profile_type in width_profiles.keys():
        experimental_perturbations[profile_type] = {
            "forcing.T0": lambda T0_spinup: T0_spinup + 1.5,  # +1.5°C warming
            "config.tf": lambda _: 500,  # 100-year response test
        }

    print(f"Width profile variations setup:")
    print(f"  Profile types: {list(width_profiles.keys())}")
    print(
        f"  Target glacier volume: 9.635e8 m³ (~0.9635 km³, neutral profile baseline)"
    )
    print(f"  Spinup duration: 500 years (reduced for testing)")
    print(f"  Response test: +1.5°C warming for 100 years")
    print(f"  Total runs: {len(spinup_objects)}")

    for profile_type, profile_info in width_profiles.items():
        print(f"  {profile_type}: {profile_info['description']}")

    # --- Run the Sweep with FlowlineSpinup Objects ---
    # Use first geometry as base (won't be used due to spinup_objects)
    base_geometry = FlowlineGeometry(
        x_gr=x_gr_top,
        zb_gr=zb_gr_top,
        w_geom=w_geom_top,
        x_init=x_gr_top,
        h_init=h_init,
    )

    sweep = FlowlineSweep(
        base_config=response_config,
        base_geometry=base_geometry,
        base_forcing=response_forcing,
        spinup_objects=spinup_objects,  # Creates runs automatically from dict keys
        experimental_perturbations=experimental_perturbations,  # Apply experimental changes
        **get_sweep_cli_kwargs(args),
    )

    sweep.run()

    print(
        f"\nWidth profile variations sweep completed. Results saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
