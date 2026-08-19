#!/usr/bin/env python3
"""
Width shape variations example with target matching.

This example compares "hourglass", "oval", and "neutral" glacier width shapes to
isolate the effect of width distribution on glacier dynamics. All shapes have the
same integral of width over [0, 8km], and the hourglass and oval are exact mirrors:

  Hourglass: wide at head/terminus (1500m), narrow at midpoint (1000m at 4km)
  Oval:      narrow at head/terminus (1000m), wide at midpoint (1500m at 4km)
  Neutral:   constant width (1250m) — arithmetic mean of hourglass/oval widths

The shapes satisfy:
  - hourglass w(0) == hourglass w(8km) == oval w(4km) == 1500m
  - oval w(0) == oval w(8km) == hourglass w(4km) == 1000m
  - equal cross-sectional area integrals over [0, 8km] (all = 10,000,000 m²)

All glaciers are adjusted to achieve the same initial volume using target matching
to optimize T0, allowing direct comparison of shape effects on glacier response.
"""

from pathlib import Path
import numpy as np

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module


def main():
    # Parse command line arguments
    args = parse_sweep_cli_args(
        "Width shape variations example: hourglass vs oval with target matching."
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

    # --- Create Width Shape Geometries ---
    # Both shapes satisfy over [0, 8km] with x_mid=4000, domain_extent=12000:
    #   w(8km) = (w_mid + w_term) / 2  [from piecewise linear geometry]
    #   Symmetry: w(0) == w(8km) → w_term = 2*w_head - w_mid
    #   Integral [0,8km] = 4000 * (w_head + w_mid) = 10,000,000 m² for both
    #
    # Hourglass: w_head=1500, w_mid=1000, w_term=2000
    #   w(0)=1500, w(4km)=1000, w(8km)=(1000+2000)/2=1500
    #
    # Oval (mirror): w_head=1000, w_mid=1500, w_term=500
    #   w(0)=1000, w(4km)=1500, w(8km)=(1500+500)/2=1000

    # Hourglass: 1500m at head, narrows to 1000m at 4km, widens back to 1500m at 8km
    x_gr_hg, zb_gr_hg, w_geom_hg = geometry_module.create_variable_width(
        w_head=1500, w_term=2000, w_mid=1000, x_mid=4000, **common_geom_params
    )

    # Oval: 1000m at head, widens to 1500m at 4km, narrows back to 1000m at 8km
    x_gr_oval, zb_gr_oval, w_geom_oval = geometry_module.create_variable_width(
        w_head=1000, w_term=500, w_mid=1500, x_mid=4000, **common_geom_params
    )

    # Neutral: constant width (1250m) — midpoint of hourglass/oval widths.
    # Integral [0,8km] = 1250 * 8000 = 10,000,000 m², matching hourglass and oval.
    x_gr_neutral, zb_gr_neutral, w_geom_neutral = geometry_module.create_uniform_slope(
        width=1250, **common_geom_params
    )

    # Store geometries for analysis
    width_profiles = {
        "hourglass": {
            "geometry_data": (x_gr_hg, zb_gr_hg, w_geom_hg),
            "description": "Wide (1500m) → narrow (1000m) at 4km → wide (1500m) at 8km",
        },
        "oval": {
            "geometry_data": (x_gr_oval, zb_gr_oval, w_geom_oval),
            "description": "Narrow (1000m) → wide (1500m) at 4km → narrow (1000m) at 8km",
        },
        "neutral": {
            "geometry_data": (x_gr_neutral, zb_gr_neutral, w_geom_neutral),
            "description": "Constant width (1250m)",
        },
    }

    # Create reasonable initial ice thickness profile for spinup
    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr_hg / length))  # Same x_gr for all profiles

    # --- Base Forcing for Response Testing ---
    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,
        T0=7.0,  # Will be overridden by spinup optimization
        mu=0.6,
    )

    # --- Create FlowlineSpinup Objects for Each Width Shape ---
    spinup_objects = {}

    for profile_type, profile_info in width_profiles.items():
        x_gr, zb_gr, w_geom = profile_info["geometry_data"]

        # Create geometry object for this width profile
        geometry = FlowlineGeometry(
            x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, h0=h_init
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

        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching={
                "targets": {
                    "target_length": 8000,  # m
                },
                "adjustment_parameters": ["T0"],  # Optimize temperature
                "bounds": [(5.5, 8.5)],  # Temperature bounds
                "cost_function": LengthOnlyCost,
                "steady_state_detector": VolumeChangeRateDetector,
                "tolerance": 50,  # Accept ±50m from target
                "max_simulation_time": 1000,
                "optimization_options": {
                    "maxfev": 25,  # Reduced function evaluations
                    "maxiter": 10,
                },
            },
        )

        spinup_objects[profile_type] = spinup_obj

    # --- Create Experimental Perturbations ---
    # Apply +1.5°C warming to test response sensitivity across width shapes
    experimental_perturbations = {}
    for profile_type in width_profiles.keys():
        experimental_perturbations[profile_type] = {
            "forcing.T0": lambda T0_spinup: T0_spinup + 0.5,  # +1.5°C warming
            "config.tf": lambda _: 500,  # 500-year response test
        }

    print("Width shape variations setup:")
    print(f"  Shape types: {list(width_profiles.keys())}")
    print("  Target glacier length: 8000m (±50m tolerance)")
    print(f"  Spinup duration: 500 years (reduced for testing)")
    print(f"  Response test: +1.5°C warming for 500 years")
    print(f"  Total runs: {len(spinup_objects)}")

    for profile_type, profile_info in width_profiles.items():
        print(f"  {profile_type}: {profile_info['description']}")

    # --- Run the Sweep with FlowlineSpinup Objects ---
    # Use first geometry as base (won't be used due to spinup_objects)
    base_geometry = FlowlineGeometry(
        x_gr=x_gr_hg,
        zb_gr=zb_gr_hg,
        w_geom=w_geom_hg,
        h0=h_init,
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
        f"\nWidth shape variations sweep completed. Results saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
