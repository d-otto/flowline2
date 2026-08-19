#!/usr/bin/env python3
"""
Width profiles flared example with target matching.

This example compares four glacier width profiles. Each profile has one flat
segment and one linearly-varying segment, with the breakpoint at x=4km. All
profiles have equal cumulative area over [0, 8km] = 10,000,000 m² (average
width 1250m). The varying segment amplitude is ±333m relative to the flat
segment.

- upper_flared:  wide at head (1500m), narrows to flat ~1167m at 4km
- upper_tapered: narrow at head (1000m), widens to flat ~1333m at 4km
- lower_flared:  flat upper reach at ~1167m, widens to 1500m at 8km
- lower_tapered: flat upper reach at ~1333m, narrows to 1000m at 8km

All glaciers are adjusted to achieve a target length of 8 km using target
matching to optimize precipitation (P0), allowing direct comparison of upper vs
lower reach width effects on glacier dynamics.
"""

from pathlib import Path
import numpy as np

from flowline.sweep import FlowlineSweep
from flowline.spinup import (
    FlowlineSpinup,
    VolumeOnlyCost,
    LengthOnlyCost,
    VolumeChangeRateDetector,
)
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module


def main():
    # Parse command line arguments
    args = parse_sweep_cli_args(
        "Width profiles flared example: flared/tapered in upper vs lower reach."
    )

    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    # --- Base Configuration for Response Testing ---
    response_config = FlowlineConfig(
        ts=0,
        tf=500,  # Response test duration: 500 years
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
    # All profiles have one flat segment and one linearly-varying segment, with breakpoint at x=4km.
    # Equal area over [0, 8km] = 10,000,000 m² (average width 1250m × 8000m).
    #
    # w_term is the width at domain_extent=12km (not at 8km). Since the second segment interpolates
    # linearly from x_mid=4km to domain_extent=12km:
    #   w(x) = w_mid + (w_term - w_mid) * (x - 4000) / 8000
    #   w(8km) = w_mid + (w_term - w_mid) / 2
    #   w_term = 2 * w(8km) - w_mid
    #
    # Design: w_wide=1500, w_narrow=1000, amplitude ±333m from flat segment.
    # Area constraint: average width = 1250m, total = 10,000,000 m²
    #
    # upper varying (flat in [4-8km]):
    #   (w_head + w_flat)/2 * 4000 + w_flat * 4000 = 10,000,000
    #   w_flat = (5000 - w_head) / 3
    #   upper_flared:  w_flat = (5000 - 1500) / 3 = 1167m
    #   upper_tapered: w_flat = (5000 - 1000) / 3 = 1333m
    #
    # lower varying (flat in [0-4km]):
    #   w_flat * 4000 + (w_flat + w_8km) / 2 * 4000 = 10,000,000
    #   w_8km = 5000 - 3 * w_flat
    #   lower_flared:  w_flat=1167, w(8km)=1500, w_term=2*1500-1167=1833
    #   lower_tapered: w_flat=1333, w(8km)=1000, w_term=2*1000-1333=667

    w_wide = 1500
    w_narrow = 1000
    w_flat_flared = (5000 - w_wide) / 3    # 1166.7m
    w_flat_tapered = (5000 - w_narrow) / 3  # 1333.3m
    w_term_lower_flared = 2 * w_wide - w_flat_flared      # 1833.3m
    w_term_lower_tapered = 2 * w_narrow - w_flat_tapered  # 666.7m

    # upper_flared: wide at head (1500m), narrows to flat 1167m at 4km, flat lower reach
    x_gr_ufl, zb_gr_ufl, w_geom_ufl = geometry_module.create_variable_width(
        w_head=w_wide, w_mid=w_flat_flared, x_mid=4000, w_term=w_flat_flared, **common_geom_params
    )

    # upper_tapered: narrow at head (1000m), widens to flat 1333m at 4km, flat lower reach
    x_gr_utp, zb_gr_utp, w_geom_utp = geometry_module.create_variable_width(
        w_head=w_narrow, w_mid=w_flat_tapered, x_mid=4000, w_term=w_flat_tapered, **common_geom_params
    )

    # lower_flared: flat upper reach at 1167m, widens from 1167m to 1500m at 8km (1833m at 12km)
    x_gr_fl, zb_gr_fl, w_geom_fl = geometry_module.create_variable_width(
        w_head=w_flat_flared, w_mid=w_flat_flared, x_mid=4000, w_term=w_term_lower_flared, **common_geom_params
    )

    # lower_tapered: flat upper reach at 1333m, narrows from 1333m to 1000m at 8km (667m at 12km)
    x_gr_tp, zb_gr_tp, w_geom_tp = geometry_module.create_variable_width(
        w_head=w_flat_tapered, w_mid=w_flat_tapered, x_mid=4000, w_term=w_term_lower_tapered, **common_geom_params
    )

    # Store geometries for loop
    width_profiles = {
        "upper_flared": {
            "geometry_data": (x_gr_ufl, zb_gr_ufl, w_geom_ufl),
            "description": f"Wide at head ({w_wide}m), narrows to flat {w_flat_flared:.0f}m at 4km",
        },
        "upper_tapered": {
            "geometry_data": (x_gr_utp, zb_gr_utp, w_geom_utp),
            "description": f"Narrow at head ({w_narrow}m), widens to flat {w_flat_tapered:.0f}m at 4km",
        },
        "lower_flared": {
            "geometry_data": (x_gr_fl, zb_gr_fl, w_geom_fl),
            "description": f"Flat upper reach at {w_flat_flared:.0f}m, widens to {w_wide}m at 8km",
        },
        "lower_tapered": {
            "geometry_data": (x_gr_tp, zb_gr_tp, w_geom_tp),
            "description": f"Flat upper reach at {w_flat_tapered:.0f}m, narrows to {w_narrow}m at 8km",
        },
    }

    # Create reasonable initial ice thickness profile for spinup
    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr_ufl / length))  # Same x_gr for all profiles

    # --- Base Forcing for Response Testing ---
    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,  # Will be overridden by spinup optimization
        T0=8.0,
        mu=0.6,
    )

    # --- Create FlowlineSpinup Objects for Each Width Profile ---
    spinup_objects = {}

    for profile_type, profile_info in width_profiles.items():
        x_gr, zb_gr, w_geom = profile_info["geometry_data"]

        geometry = FlowlineGeometry(
            x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, h0=h_init
        )

        spinup_config = FlowlineConfig(
            ts=0,
            tf=1000,  # 1000-year spinup
            delx=25,
            delt=0.00078125,
            deltout=1.0,
            min_thick=1.0,
        )

        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0,
            tf=1000,
            P0=2.0,  # Will be adjusted by target matching
            T0=8.0,
            mu=0.6,
        )

        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching={
                "targets": {
                    "target_length": 8000,  # 1 km³ in m³
                },
                "adjustment_parameters": ["P0"],  # Optimize precipitation
                "bounds": [(0.0, 4.0)],  # Precipitation bounds
                "cost_function": LengthOnlyCost,
                "steady_state_detector": VolumeChangeRateDetector,
                "tolerance": 50,  # Accept ±10 million m³ (±0.01 km³) from target
                "max_simulation_time": 1000,
                "optimization_options": {
                    "maxfev": 15,
                    "maxiter": 10,
                },
            },
        )

        spinup_objects[profile_type] = spinup_obj

    # --- Create Experimental Perturbations ---
    # Apply +1°C warming to test response sensitivity across width profiles
    experimental_perturbations = {}
    for profile_type in width_profiles.keys():
        experimental_perturbations[profile_type] = {
            "forcing.T0": lambda T0_spinup: T0_spinup + 1.0,  # +1°C warming
            "config.tf": lambda _: 500,  # 500-year response test
        }

    print("Width profiles flared setup:")
    print(f"  Profile types: {list(width_profiles.keys())}")
    print("  Target glacier length: 8000m (±50m tolerance)")
    print("  Adjustment parameter: P0 (precipitation), bounds [0, 4] m/yr")
    print(f"  Width design: breakpoint at 4km; equal [0-8km] area (10,000,000 m², avg 1250m) for all profiles")
    print(f"  w_wide={w_wide}, w_narrow={w_narrow}, w_flat_flared={w_flat_flared:.1f}, w_flat_tapered={w_flat_tapered:.1f}")
    print("  Spinup duration: 1000 years")
    print("  Response test: +1°C warming for 500 years")
    print(f"  Total runs: {len(spinup_objects)}")

    for profile_type, profile_info in width_profiles.items():
        print(f"  {profile_type}: {profile_info['description']}")

    # --- Run the Sweep with FlowlineSpinup Objects ---
    base_geometry = FlowlineGeometry(
        x_gr=x_gr_ufl,
        zb_gr=zb_gr_ufl,
        w_geom=w_geom_ufl,
        h0=h_init,
    )

    sweep = FlowlineSweep(
        base_config=response_config,
        base_geometry=base_geometry,
        base_forcing=response_forcing,
        spinup_objects=spinup_objects,
        experimental_perturbations=experimental_perturbations,
        **get_sweep_cli_kwargs(args),
    )

    sweep.run()

    print(
        f"\nWidth profiles flared sweep completed. Results saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
