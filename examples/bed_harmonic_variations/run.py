#!/usr/bin/env python3
"""
Bed harmonic variations example with target matching.

Analogous to width_harmonic_variations, but varies bed shape instead of width.
All three glaciers share the same constant width (1250m) and harmonic width profiles:
  - Hourglass width + convex bed: wide at head/terminus, narrow at center; bed humped at center
  - Oval width    + concave bed: narrow at head/terminus, wide at center; bed overdeepened at center
  - Neutral: constant width, flat uniform slope

Bed perturbation shape: A * sin(pi * x / L)^2
  Positive A -> convex (elevated center)
  Negative A -> concave (overdeepened center)

Analysis section (after spinup):
  Loads width_harmonic_variations spinup profiles and computes, for each non-neutral case,
  the integral over [0, 8km] of (h_neutral - h_case). Then finds the bed perturbation
  amplitude A such that the integral of the bed perturbation over [0, 8km] matches that
  thickness-difference integral. Since integral_0^L sin^2(pi*x/L) dx = L/2, the
  analytic solution is A = 2 * thickness_diff_integral / L.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_harmonic_width

HERE = Path(__file__).resolve().parent
WIDTH_HARMONIC_OUTPUT = HERE.parent / "width_harmonic_variations" / "output"


def compute_bed_amplitude_from_width_variations(L=8000.0):
    """
    Load width_harmonic_variations spinup profiles and return matched bed amplitudes.

    For each non-neutral case, computes:
        thickness_diff_integral = integral_0^L (h_neutral - h_case) dx

    Then finds A such that:
        integral_0^L A * sin^2(pi*x/L) dx = thickness_diff_integral
        A = 2 * thickness_diff_integral / L

    Returns
    -------
    dict with keys 'hourglass' and 'oval', values are bed perturbation amplitudes in metres.
    Hourglass amplitude is positive (convex bed), oval amplitude is negative (concave bed).
    """
    spinup_dir = WIDTH_HARMONIC_OUTPUT / "spinup_profiles"
    profiles = {}
    for case in ("neutral", "hourglass", "oval"):
        path = spinup_dir / f"spinup_spinup_{case}.nc"
        with xr.open_dataset(path) as ds:
            profiles[case] = {
                "h": ds["h"].isel(time=-1).values,
                "x": ds["x"].values,
            }

    x = profiles["neutral"]["x"]
    mask = x <= L
    x_masked = x[mask]

    h_neutral = profiles["neutral"]["h"][mask]

    amplitudes = {}
    for case in ("hourglass", "oval"):
        h_case = profiles[case]["h"][mask]
        diff = h_neutral - h_case
        thickness_diff_integral = np.trapezoid(diff, x_masked)
        A = 2.0 * thickness_diff_integral / L
        amplitudes[case] = A

    return amplitudes


def plot_bed_amplitude_analysis(amplitudes, L=8000.0, output_dir=None):
    x = np.linspace(0, L, 500)
    perturbation_shape = np.sin(np.pi * x / L) ** 2

    spinup_dir = WIDTH_HARMONIC_OUTPUT / "spinup_profiles"
    profiles = {}
    for case in ("neutral", "hourglass", "oval"):
        path = spinup_dir / f"spinup_spinup_{case}.nc"
        with xr.open_dataset(path) as ds:
            profiles[case] = {
                "h": ds["h"].isel(time=-1).values,
                "x": ds["x"].values,
            }

    x_data = profiles["neutral"]["x"]
    mask = x_data <= L
    x_masked = x_data[mask]

    mosaic = [
        ["thickness_diff", "bed_perturbation"],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 5))

    ax = axes["thickness_diff"]
    h_neutral = profiles["neutral"]["h"][mask]
    for case, color in (("hourglass", "tab:blue"), ("oval", "tab:orange")):
        h_case = profiles[case]["h"][mask]
        diff = h_neutral - h_case
        integral = np.trapezoid(diff, x_masked)
        ax.plot(
            x_masked / 1000,
            diff,
            color=color,
            label=f"{case} (integral={integral:.0f} m²)",
        )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Distance from head [km]")
    ax.set_ylabel("h_neutral - h_case [m]")
    ax.set_title("Thickness difference: neutral minus case (width_harmonic_variations)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes["bed_perturbation"]
    for case, color in (("hourglass", "tab:blue"), ("oval", "tab:orange")):
        A = amplitudes[case]
        bed_perturb = A * perturbation_shape
        integral = np.trapezoid(bed_perturb, x)
        ax.plot(
            x / 1000,
            bed_perturb,
            color=color,
            label=f"{case}: A={A:.1f} m (integral={integral:.0f} m²)",
        )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Distance from head [km]")
    ax.set_ylabel("Bed perturbation [m]")
    ax.set_title("Matched bed perturbation A * sin²(πx/L)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Bed amplitudes matched to width_harmonic_variations thickness differences",
        fontsize=12,
    )
    plt.tight_layout()

    if output_dir is not None:
        fig.savefig(Path(output_dir) / "bed_amplitude_analysis.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_sweep_cli_args(
        "Bed harmonic variations: convex bed for hourglass width, concave bed for oval width."
    )

    if args.output_dir is None:
        args.output_dir = str(HERE / "output")

    # --- Analysis section: match bed amplitudes to width_harmonic_variations thickness diffs ---
    print("Computing bed amplitudes from width_harmonic_variations spinup profiles...")
    amplitudes = compute_bed_amplitude_from_width_variations(L=8000.0)
    amplitudes = {case: A * 3 for case, A in amplitudes.items()}
    for case, A in amplitudes.items():
        sign = "convex" if A > 0 else "concave"
        print(f"  {case}: A = {A:.2f} m ({sign})")
    plot_bed_amplitude_analysis(amplitudes, output_dir=args.output_dir)
    print(f"  Analysis plot saved to {args.output_dir}/bed_amplitude_analysis.png")

    # --- Simulation setup ---
    response_config = FlowlineConfig(
        ts=0,
        tf=100,
        delx=25,
        delt=0.00078125,
        deltout=1.0,
        min_thick=1.0,
    )

    common_geom_params = dict(
        domain_extent=12000,
        x_gr_points=61,
        elevation_drop=1000,
        bed_characteristic_length=8000,
    )

    offset = 1250
    R = 750

    # Width profiles: same as width_harmonic_variations
    # Bed shape: hourglass gets convex bed, oval gets concave bed
    bed_configs = {
        "hourglass": {
            "width_harmonics": [(1, R, np.pi)],
            "bed_perturbation": amplitudes["hourglass"],
            "description": "Hourglass width + convex bed (bump at center)",
        },
        "oval": {
            "width_harmonics": [(1, R, 0)],
            "bed_perturbation": amplitudes["oval"],
            "description": "Oval width + concave bed (overdeepening at center)",
        },
        "neutral": {
            "width_harmonics": [],
            "bed_perturbation": 0.0,
            "description": "Constant width, flat uniform slope",
        },
    }

    x_gr_ref = np.linspace(0, common_geom_params["domain_extent"], common_geom_params["x_gr_points"])
    h_init = np.maximum(0, 100 * (1 - x_gr_ref / 5000))

    response_forcing = TemperaturePrecipitationForcing(
        ts=response_config.ts,
        tf=response_config.tf,
        P0=2.0,
        T0=7.0,
        mu=0.6,
    )

    geom_data = {}
    for case, cfg in bed_configs.items():
        if cfg["bed_perturbation"] == 0.0 and not cfg["width_harmonics"]:
            # Neutral: flat bed, constant width
            x_gr, zb_gr, w_geom = create_harmonic_width(
                harmonics=[],
                offset=offset,
                **common_geom_params,
            )
        else:
            # Build width profile from harmonics, then apply bed perturbation
            x_gr, zb_gr, w_geom = create_harmonic_width(
                harmonics=cfg["width_harmonics"],
                offset=offset,
                **common_geom_params,
            )
            L = common_geom_params["bed_characteristic_length"]
            zb_gr = zb_gr + cfg["bed_perturbation"] * np.sin(np.pi * x_gr / L) ** 2
        geom_data[case] = (x_gr, zb_gr, w_geom)

    spinup_objects = {}
    for case, cfg in bed_configs.items():
        x_gr, zb_gr, w_geom = geom_data[case]
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
        spinup_objects[case] = spinup_obj

    experimental_perturbations = {}
    for case in bed_configs:
        experimental_perturbations[case] = {
            "forcing.T0": lambda T0_spinup: T0_spinup + 0.5,
            "config.tf": lambda _: 500,
        }

    print("\nBed harmonic variations setup:")
    print(f"  Cases: {list(bed_configs.keys())}")
    print(f"  Width offset={offset} m, R={R} m -> min={offset - R} m, max={offset + R} m")
    for case, cfg in bed_configs.items():
        print(f"  {case}: {cfg['description']}")

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
