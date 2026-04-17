#!/usr/bin/env python3
"""
Width profile equilibrium example: finding the per-profile warming required to
reach a prescribed retreat target.

This example answers the question: given that glaciers with different width profiles
all start at the same equilibrium length (8000m), what temperature increase is needed
for each profile to reach the same retreated equilibrium length (6000m)?

Two sequential target-matching phases are run:

  Phase 1: Optimize T0 per width profile to reach initial equilibrium at 8000m.
  Phase 2: Optimize T0 per width profile to reach retreated equilibrium at 6000m,
           using T0_phase1 as the lower bound of the search (retreat requires warming).

The key output is delta_T = T0_phase2 - T0_phase1 per profile, which reveals how
glacier geometry controls the temperature sensitivity needed to drive a fixed retreat.

Width profiles (same as width_profile_variations):
- Top-heavy: Wide at head (1750m), narrow at terminus (500m)
- Bottom-heavy: Narrow at head (750m), wide at terminus (2000m)
- Neutral: Constant width (1250m)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowline.spinup import (
    FlowlineSpinup,
    VolumeChangeRateDetector,
    LengthOnlyCost,
)
from flowline.cli.utils import parse_sweep_cli_args
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry_module


TARGET_LENGTH_PHASE1 = 8000  # m — initial equilibrium target
TARGET_LENGTH_PHASE2 = 6000  # m — retreated equilibrium target


def run_target_matching_phase(profile_type, x_gr, zb_gr, w_geom, h_init, T0_init, target_length, T0_bounds, output_dir, no_progress):
    """Run one target-matching spinup and return (profile_path, T0_optimized)."""
    spinup_obj = FlowlineSpinup(
        config=FlowlineConfig(
            ts=0,
            tf=500,
            delx=25,
            delt=0.00078125,
            deltout=1.0,
            min_thick=1.0,
        ),
        geometry=FlowlineGeometry(
            x_gr=x_gr,
            zb_gr=zb_gr,
            w_geom=w_geom,
            x_init=x_gr,
            h_init=h_init,
        ),
        forcing=TemperaturePrecipitationForcing(
            ts=0,
            tf=1000,
            P0=2.0,
            T0=T0_init,
            mu=0.6,
        ),
        target_matching={
            "targets": {"target_length": target_length},
            "adjustment_parameters": ["T0"],
            "bounds": [T0_bounds],
            "cost_function": LengthOnlyCost,
            "steady_state_detector": VolumeChangeRateDetector,
            "tolerance": 50,
            "max_simulation_time": 1000,
            "optimization_options": {"maxfev": 25, "maxiter": 10},
        },
    )
    profile_path, optimized = spinup_obj.generate_profile(output_dir, profile_type, no_progress)
    return profile_path, optimized["T0"]


def plot_results(width_profiles, phase1_results, phase2_results, output_dir):
    """Plot ΔT per profile and T0 comparison between phases."""
    profile_names = list(width_profiles.keys())
    T0_p1 = [phase1_results[p]["T0"] for p in profile_names]
    T0_p2 = [phase2_results[p]["T0"] for p in profile_names]
    delta_T = [T0_p2[i] - T0_p1[i] for i in range(len(profile_names))]

    fig, axes = plt.subplot_mosaic([["delta_T", "T0_comparison"]], figsize=(12, 5))

    # Panel A: ΔT per profile
    ax = axes["delta_T"]
    x_pos = range(len(profile_names))
    ax.scatter(x_pos, delta_T, s=100, zorder=3)
    for i, (name, dT) in enumerate(zip(profile_names, delta_T)):
        ax.annotate(f"{dT:+.3f}°C", (i, dT), textcoords="offset points", xytext=(8, 0), va="center")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(profile_names, rotation=15, ha="right")
    ax.set_ylabel("ΔT (°C)")
    ax.set_title(f"Warming required to retreat from {TARGET_LENGTH_PHASE1/1000:.0f}km to {TARGET_LENGTH_PHASE2/1000:.0f}km")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)

    # Panel B: T0_phase1 and T0_phase2 per profile
    ax = axes["T0_comparison"]
    for i, name in enumerate(profile_names):
        ax.plot([i - 0.1, i + 0.1], [T0_p1[i], T0_p2[i]], "k-", linewidth=1, zorder=2)
        ax.scatter([i - 0.1], [T0_p1[i]], s=80, label="Phase 1 (8km)" if i == 0 else None, zorder=3)
        ax.scatter([i + 0.1], [T0_p2[i]], s=80, marker="s", label="Phase 2 (6km)" if i == 0 else None, zorder=3)
    ax.set_xticks(list(range(len(profile_names))))
    ax.set_xticklabels(profile_names, rotation=15, ha="right")
    ax.set_ylabel("T0 (°C)")
    ax.set_title("Equilibrium temperature by phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = Path(output_dir) / "width_profile_equilibrium_results.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Results plot saved to: {plot_path}")


def main():
    args = parse_sweep_cli_args(
        "Width profile equilibrium: find per-profile warming to reach a prescribed retreat target."
    )

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    # --- Define Common Geometry Parameters ---
    common_geom_params = {
        "domain_extent": 12000,
        "x_gr_points": 61,
        "elevation_drop": 1000,
        "bed_characteristic_length": 10000,
    }

    # --- Create Width Profile Geometries (identical to width_profile_variations) ---
    x_gr_top, zb_gr_top, w_geom_top = geometry_module.create_variable_width(
        w_head=1750, w_term=500, **common_geom_params
    )
    x_gr_bottom, zb_gr_bottom, w_geom_bottom = geometry_module.create_variable_width(
        w_head=750, w_term=2000, **common_geom_params
    )
    x_gr_neutral, zb_gr_neutral, w_geom_neutral = geometry_module.create_uniform_slope(
        width=1250, **common_geom_params
    )

    width_profiles = {
        "top_heavy": (x_gr_top, zb_gr_top, w_geom_top),
        "bottom_heavy": (x_gr_bottom, zb_gr_bottom, w_geom_bottom),
        "neutral": (x_gr_neutral, zb_gr_neutral, w_geom_neutral),
    }

    # Initial ice thickness profile (same for all profiles)
    h_init = np.maximum(0, 100 * (1 - x_gr_top / 5000))

    # --- Phase 1: Find T0 for initial equilibrium at TARGET_LENGTH_PHASE1 ---
    phase1_output_dir = Path(args.output_dir) / "phase1"
    phase1_results = {}

    print(f"\n=== Phase 1: Target matching to {TARGET_LENGTH_PHASE1}m equilibrium ===")
    for profile_type, (x_gr, zb_gr, w_geom) in width_profiles.items():
        print(f"\n  Running Phase 1 for '{profile_type}'...")
        profile_path, T0_opt = run_target_matching_phase(
            profile_type=profile_type,
            x_gr=x_gr,
            zb_gr=zb_gr,
            w_geom=w_geom,
            h_init=h_init,
            T0_init=7.0,
            target_length=TARGET_LENGTH_PHASE1,
            T0_bounds=(5.5, 8.5),
            output_dir=phase1_output_dir,
            no_progress=args.no_progress,
        )
        phase1_results[profile_type] = {"T0": T0_opt, "profile_path": profile_path}
        print(f"  Phase 1 [{profile_type}]: T0 = {T0_opt:.3f}°C")

    # --- Phase 2: Find T0 for retreated equilibrium at TARGET_LENGTH_PHASE2 ---
    # Bounds derived from Phase 1: lower bound = T0_phase1 (retreat requires warming)
    phase2_output_dir = Path(args.output_dir) / "phase2"
    phase2_results = {}

    print(f"\n=== Phase 2: Target matching to {TARGET_LENGTH_PHASE2}m equilibrium ===")
    for profile_type, (x_gr, zb_gr, w_geom) in width_profiles.items():
        T0_phase1 = phase1_results[profile_type]["T0"]
        print(f"\n  Running Phase 2 for '{profile_type}' (T0 lower bound: {T0_phase1:.3f}°C)...")
        profile_path, T0_opt = run_target_matching_phase(
            profile_type=profile_type,
            x_gr=x_gr,
            zb_gr=zb_gr,
            w_geom=w_geom,
            h_init=h_init,
            T0_init=T0_phase1,
            target_length=TARGET_LENGTH_PHASE2,
            T0_bounds=(T0_phase1, T0_phase1 + 2.0),
            output_dir=phase2_output_dir,
            no_progress=args.no_progress,
        )
        phase2_results[profile_type] = {"T0": T0_opt, "profile_path": profile_path}
        print(f"  Phase 2 [{profile_type}]: T0 = {T0_opt:.3f}°C")

    # --- Summary ---
    print(f"\n=== Results: ΔT required to retreat from {TARGET_LENGTH_PHASE1/1000:.0f}km to {TARGET_LENGTH_PHASE2/1000:.0f}km ===")
    print(f"{'Profile':<15} {'T0_phase1 (°C)':>16} {'T0_phase2 (°C)':>16} {'ΔT (°C)':>10}")
    print("-" * 60)
    for profile_type in width_profiles:
        T0_p1 = phase1_results[profile_type]["T0"]
        T0_p2 = phase2_results[profile_type]["T0"]
        delta_T = T0_p2 - T0_p1
        print(f"{profile_type:<15} {T0_p1:>16.3f} {T0_p2:>16.3f} {delta_T:>+10.3f}")

    plot_results(width_profiles, phase1_results, phase2_results, args.output_dir)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
