#!/usr/bin/env python3
"""
Width harmonic variations example with target matching.

Compares "hourglass", "oval", and "neutral" glacier width shapes defined by
cosine harmonics rather than piecewise linear segments. Analogous to the
heavyside_functions.ipynb framework: each shape is a sum of orthogonal
harmonic components, making it straightforward to modify the profile by
adding or changing harmonics.

Width formula:
    w(x) = offset + sum_i( R_i * cos(n_i * 2*pi*x/L + phi_i) )

where L = bed_characteristic_length (one full period = glacier length).
Because each cosine integrates to zero over a full period, the mean width
equals `offset` for all shapes, so equal-area integrals are automatic.

Profiles (offset=1250, R=750):
  Hourglass: phi=0   -> wide (2000m) at head/terminus, narrow (500m) at center
  Oval:      phi=pi  -> narrow (500m) at head/terminus, wide (2000m) at center
  Neutral:   no harmonics -> constant 1250m (the mean of hourglass/oval)
"""

from pathlib import Path
import numpy as np

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_harmonic_width


def main():
    args = parse_sweep_cli_args(
        "Width harmonic variations: hourglass vs oval using cosine harmonic profiles."
    )

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    response_config = FlowlineConfig(
        ts=0,
        tf=100,
        delx=25,
        delt=0.00078125,
        deltout=1.0,
        min_thick=1.0,
    )

    common_geom_params = {
        "domain_extent": 12000,
        "x_gr_points": 61,
        "elevation_drop": 1000,
        "bed_characteristic_length": 8000,
    }

    # offset=1250 sets the mean width; R=750 gives min=500m, max=2000m.
    # All three profiles have the same cross-sectional area integral over [0, 8km]
    # because each cosine harmonic integrates to zero over one full period.
    offset = 1250
    R = 750

    width_profiles = {
        "hourglass": {
            "harmonics": [(1, R, 0)],
            "description": "Wide (2000m) at head/terminus, narrow (500m) at center",
        },
        "oval": {
            "harmonics": [(1, R, np.pi)],
            "description": "Narrow (500m) at head/terminus, wide (2000m) at center",
        },
        "neutral": {
            "harmonics": [],
            "description": "Constant width (1250m) — mean of hourglass/oval",
        },
    }

    # Build geometry arrays for each profile
    geom_data = {}
    for profile_type, profile_info in width_profiles.items():
        x_gr, zb_gr, w_geom = create_harmonic_width(
            harmonics=profile_info["harmonics"],
            offset=offset,
            **common_geom_params,
        )
        geom_data[profile_type] = (x_gr, zb_gr, w_geom)

    # Initial ice thickness (same x_gr for all profiles)
    x_gr_ref = geom_data["hourglass"][0]
    h_init = np.maximum(0, 100 * (1 - x_gr_ref / 5000))

    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,
        T0=7.0,
        mu=0.6,
    )

    spinup_objects = {}
    for profile_type, profile_info in width_profiles.items():
        x_gr, zb_gr, w_geom = geom_data[profile_type]

        geometry = FlowlineGeometry(x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, h0=h_init)

        spinup_config = FlowlineConfig(
            ts=0,
            tf=500,
            delx=25,
            delt=0.00078125,
            deltout=1.0,
            min_thick=1.0,
        )

        spinup_forcing = TemperaturePrecipitationForcing(
            ts=0,
            tf=1000,
            P0=2.0,
            T0=6.6,
            mu=0.6,
        )

        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=spinup_forcing,
            target_matching={
                "targets": {"target_length": 8000},
                "adjustment_parameters": ["T0"],
                "bounds": [(5.5, 7.5)],
                "cost_function": LengthOnlyCost,
                "steady_state_detector": VolumeChangeRateDetector,
                "tolerance": 50,
                "max_simulation_time": 1000,
                "optimization_options": {"maxfev": 25, "maxiter": 10},
            },
        )

        spinup_objects[profile_type] = spinup_obj

    experimental_perturbations = {}
    for profile_type in width_profiles:
        experimental_perturbations[profile_type] = {
            "forcing.T0": lambda T0_spinup: T0_spinup + 0.5,
            "config.tf": lambda _: 500,
        }

    print("Width harmonic variations setup:")
    print(f"  Shape types: {list(width_profiles.keys())}")
    print(f"  offset={offset} m, R={R} m -> min={offset - R} m, max={offset + R} m")
    print("  Target glacier length: 8000m (+/-50m tolerance)")
    print("  Spinup duration: 500 years")
    print("  Response test: +2C warming for 500 years")
    for profile_type, profile_info in width_profiles.items():
        print(f"  {profile_type}: {profile_info['description']}")

    base_geom_x, base_geom_zb, base_geom_w = geom_data["hourglass"]
    base_geometry = FlowlineGeometry(
        x_gr=base_geom_x,
        zb_gr=base_geom_zb,
        w_geom=base_geom_w,
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

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
