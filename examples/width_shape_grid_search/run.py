#!/usr/bin/env python3
"""
Width shape grid search: temperature sensitivity of glaciers with different width shapes.

For each width shape (hourglass, oval, neutral), the glacier is spun up to the same
equilibrium length (8000m) via target matching, then a grid of temperature perturbations
(+/-1 deg C in 0.25 deg steps) is applied to map out how each shape responds to
temperature forcing.

Shapes:
  Hourglass: wide at head/terminus (1500m), narrow at midpoint (1000m at 4km)
  Oval:      narrow at head/terminus (1000m), wide at midpoint (1500m at 4km)
  Neutral:   constant width (1250m)
"""

from pathlib import Path
import numpy as np

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup, VolumeChangeRateDetector, LengthOnlyCost
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module


TARGET_LENGTH = 8000  # m — spinup equilibrium target for all shapes
DT_VALUES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # deg C


def make_run_id(shape: str, dT: float) -> str:
    sign = "p" if dT >= 0 else "m"
    return f"{shape}_dT_{sign}{abs(dT):.2f}"


def main():
    args = parse_sweep_cli_args(
        "Width shape grid search: temperature sensitivity across width shapes."
    )

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    # --- Common geometry parameters (identical to width_shape_variations) ---
    common_geom_params = {
        "domain_extent": 12000,
        "x_gr_points": 61,
        "elevation_drop": 1000,
        "bed_characteristic_length": 10000,
    }

    # Hourglass: 1500m at head, narrows to 1000m at 4km, widens back to 1500m at 8km
    x_gr_hg, zb_gr_hg, w_geom_hg = geometry_module.create_variable_width(
        w_head=1500, w_term=2000, w_mid=1000, x_mid=4000, **common_geom_params
    )

    # Oval: 1000m at head, widens to 1500m at 4km, narrows back to 1000m at 8km
    x_gr_oval, zb_gr_oval, w_geom_oval = geometry_module.create_variable_width(
        w_head=1000, w_term=500, w_mid=1500, x_mid=4000, **common_geom_params
    )

    # Neutral: constant width (1250m)
    x_gr_neutral, zb_gr_neutral, w_geom_neutral = geometry_module.create_uniform_slope(
        width=1250, **common_geom_params
    )

    width_shapes = {
        "hourglass": (x_gr_hg, zb_gr_hg, w_geom_hg),
        "oval": (x_gr_oval, zb_gr_oval, w_geom_oval),
        "neutral": (x_gr_neutral, zb_gr_neutral, w_geom_neutral),
    }

    h_init = np.maximum(0, 100 * (1 - x_gr_hg / 5000))

    # --- Create one FlowlineSpinup per width shape ---
    shape_spinups = {}
    for shape_name, (x_gr, zb_gr, w_geom) in width_shapes.items():
        shape_spinups[shape_name] = FlowlineSpinup(
            config=FlowlineConfig(
                ts=0,
                tf=500,
                delx=25,
                delt=0.00078125,
                deltout=1.0,
                min_thick=1.0,
            ),
            geometry=FlowlineGeometry(x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, h0=h_init),
            forcing=TemperaturePrecipitationForcing(
                ts=0,
                tf=1000,
                P0=2.0,
                T0=7.0,
                mu=0.6,
            ),
            target_matching={
                "targets": {"target_length": TARGET_LENGTH},
                "adjustment_parameters": ["T0"],
                "bounds": [(5.5, 8.5)],
                "cost_function": LengthOnlyCost,
                "steady_state_detector": VolumeChangeRateDetector,
                "tolerance": 50,
                "max_simulation_time": 1000,
                "optimization_options": {"maxfev": 25, "maxiter": 10},
            },
        )

    # --- Build 27-run sweep: 9 dT values x 3 shapes ---
    # Reuse spinup objects across temperature runs; deduplication in FlowlineSweep
    # ensures each unique spinup runs only once.
    spinup_objects = {}
    experimental_perturbations = {}

    for shape_name, spinup_obj in shape_spinups.items():
        for dT in DT_VALUES:
            run_id = make_run_id(shape_name, dT)
            spinup_objects[run_id] = spinup_obj
            experimental_perturbations[run_id] = {
                "forcing.T0": lambda T0, dT=dT: T0 + dT,
                "config.tf": lambda _: 500,
            }

    # --- Response run configuration ---
    response_config = FlowlineConfig(
        ts=0,
        tf=500,
        delx=25,
        delt=0.00078125,
        deltout=1.0,
        min_thick=1.0,
    )
    response_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=500,
        P0=2.0,
        T0=7.0,
        mu=0.6,
    )
    base_geometry = FlowlineGeometry(
        x_gr=x_gr_hg, zb_gr=zb_gr_hg, w_geom=w_geom_hg, h0=h_init
    )

    print(f"Grid search: {len(DT_VALUES)} temperature offsets x {len(width_shapes)} shapes = {len(spinup_objects)} runs")
    print(f"dT values: {DT_VALUES}")
    print(f"Shapes: {list(width_shapes.keys())}")

    sweep = FlowlineSweep(
        base_config=response_config,
        base_geometry=base_geometry,
        base_forcing=response_forcing,
        spinup_objects=spinup_objects,
        experimental_perturbations=experimental_perturbations,
        **get_sweep_cli_kwargs(args),
    )
    sweep.run()

    print(f"\nGrid search sweep completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
