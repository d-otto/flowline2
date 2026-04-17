#!/usr/bin/env python3
"""
Visualization for the width_profile_equilibrium example.

Reads Phase 1 and Phase 2 spinup profiles and produces the analysis plot.
Can be run standalone: uv run examples/width_profile_equilibrium/plot.py [output_dir]
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


PROFILE_TYPES = ["top_heavy", "bottom_heavy", "neutral"]
COLORS = {
    "top_heavy": "#1565C0",
    "bottom_heavy": "#B71C1C",
    "neutral": "#2E7D32",
}


def load_spinup_profiles(phase_dir):
    """Load all three spinup profile datasets from a phase output directory."""
    spinup_dir = phase_dir / "spinup_profiles"
    profiles = {}
    for profile_type in PROFILE_TYPES:
        path = spinup_dir / f"spinup_spinup_{profile_type}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing spinup profile: {path}")
        profiles[profile_type] = xr.open_dataset(path)
    return profiles


def plot(output_dir):
    output_dir = Path(output_dir)
    phase1_dir = output_dir / "phase1"
    phase2_dir = output_dir / "phase2"

    for d in (phase1_dir, phase2_dir):
        if not d.exists():
            raise FileNotFoundError(f"Phase output directory not found: {d}")

    print("Loading spinup profiles...")
    p1 = load_spinup_profiles(phase1_dir)
    p2 = load_spinup_profiles(phase2_dir)

    # Extract key scalars
    T0_p1 = {pt: float(ds.attrs["forcing_T0"]) for pt, ds in p1.items()}
    T0_p2 = {pt: float(ds.attrs["forcing_T0"]) for pt, ds in p2.items()}
    delta_T = {pt: T0_p2[pt] - T0_p1[pt] for pt in PROFILE_TYPES}

    def final_length(ds):
        delx = float(ds.attrs["delx"])
        return float(ds["edge"].isel(time=-1))

    def final_volume_km3(ds):
        delx = float(ds.attrs["delx"])
        return float((ds["h"].isel(time=-1) * ds["w"] * delx).sum()) / 1e9

    len_p1 = {pt: final_length(ds) for pt, ds in p1.items()}
    len_p2 = {pt: final_length(ds) for pt, ds in p2.items()}
    vol_p1 = {pt: final_volume_km3(ds) for pt, ds in p1.items()}
    vol_p2 = {pt: final_volume_km3(ds) for pt, ds in p2.items()}

    # --- Layout ---
    fig, axes = plt.subplot_mosaic(
        [
            ["width_profiles", "thickness_p1",  "thickness_p2",  "delta_T"      ],
            ["ela_p1",         "ela_p2",         "length_spinup", "volume_spinup"],
        ],
        figsize=(20, 10),
        layout="constrained",
    )
    fig.suptitle(
        "Width Profile Equilibrium: Per-Profile Warming to Drive 8km → 6km Retreat",
        fontsize=13,
    )

    # 1. Width profiles
    ax = axes["width_profiles"]
    for pt in PROFILE_TYPES:
        ds = p1[pt]
        ax.plot(
            ds.coords["x"] / 1000,
            ds["w_geom_resampled"],
            color=COLORS[pt],
            linewidth=2,
            label=pt.replace("_", " ").title(),
        )
    ax.set_xlabel("Distance from head (km)")
    ax.set_ylabel("Width (m)")
    ax.set_title("Width Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Final ice thickness — Phase 1 (8km equilibrium)
    ax = axes["thickness_p1"]
    for pt in PROFILE_TYPES:
        ds = p1[pt]
        h_final = ds["h"].isel(time=-1)
        zb = ds["zb"]
        ax.fill_between(
            ds.coords["x"] / 1000,
            zb,
            zb + h_final,
            alpha=0.35,
            color=COLORS[pt],
        )
        ax.plot(ds.coords["x"] / 1000, zb + h_final, color=COLORS[pt], linewidth=1.5,
                label=f"{pt.replace('_',' ').title()} ({len_p1[pt]/1000:.1f}km)")
    ax.plot(ds.coords["x"] / 1000, ds["zb"], "k-", linewidth=1.5, label="Bed")
    ax.set_xlabel("Distance from head (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Phase 1 Equilibrium Profiles (target: 8km)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Final ice thickness — Phase 2 (6km equilibrium)
    ax = axes["thickness_p2"]
    for pt in PROFILE_TYPES:
        ds = p2[pt]
        h_final = ds["h"].isel(time=-1)
        zb = ds["zb"]
        ax.fill_between(
            ds.coords["x"] / 1000,
            zb,
            zb + h_final,
            alpha=0.35,
            color=COLORS[pt],
        )
        ax.plot(ds.coords["x"] / 1000, zb + h_final, color=COLORS[pt], linewidth=1.5,
                label=f"{pt.replace('_',' ').title()} ({len_p2[pt]/1000:.1f}km)")
    ax.plot(ds.coords["x"] / 1000, ds["zb"], "k-", linewidth=1.5, label="Bed")
    ax.set_xlabel("Distance from head (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Phase 2 Equilibrium Profiles (target: 6km)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. ΔT per profile
    ax = axes["delta_T"]
    x_pos = np.arange(len(PROFILE_TYPES))
    for i, pt in enumerate(PROFILE_TYPES):
        ax.scatter(i, delta_T[pt], color=COLORS[pt], s=120, zorder=3)
        ax.annotate(
            f"{delta_T[pt]:+.3f}°C",
            (i, delta_T[pt]),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            fontsize=9,
        )
    # T0 values as secondary reference
    for i, pt in enumerate(PROFILE_TYPES):
        ax.plot(
            [i - 0.15, i + 0.15],
            [T0_p1[pt], T0_p2[pt]],
            color=COLORS[pt],
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pt.replace("_", " ").title() for pt in PROFILE_TYPES], rotation=12)
    ax.set_ylabel("ΔT required (°C)")
    ax.set_title("Warming Required per Profile\n(8km → 6km retreat)")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)

    # 5. ELA over spinup time — Phase 1
    ax = axes["ela_p1"]
    for pt in PROFILE_TYPES:
        ds = p1[pt]
        if "ela" in ds:
            ax.plot(ds.coords["time"], ds["ela"], color=COLORS[pt], linewidth=1.5,
                    label=pt.replace("_", " ").title())
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("ELA (m)")
    ax.set_title("ELA During Phase 1 Spinup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. ELA over spinup time — Phase 2
    ax = axes["ela_p2"]
    for pt in PROFILE_TYPES:
        ds = p2[pt]
        if "ela" in ds:
            ax.plot(ds.coords["time"], ds["ela"], color=COLORS[pt], linewidth=1.5,
                    label=pt.replace("_", " ").title())
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("ELA (m)")
    ax.set_title("ELA During Phase 2 Spinup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. Glacier length over time — both phases
    ax = axes["length_spinup"]
    for pt in PROFILE_TYPES:
        ds1 = p1[pt]
        ds2 = p2[pt]
        t_offset = float(ds1.coords["time"].max())
        ax.plot(ds1.coords["time"], ds1["edge"] / 1000,
                color=COLORS[pt], linewidth=1.5, label=pt.replace("_", " ").title())
        ax.plot(ds2.coords["time"] + t_offset, ds2["edge"] / 1000,
                color=COLORS[pt], linewidth=1.5, linestyle="--")
    ax.axvline(t_offset, color="k", linewidth=0.8, linestyle=":", alpha=0.6, label="Phase boundary")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Glacier length (km)")
    ax.set_title("Glacier Length: Phase 1 (solid) / Phase 2 (dashed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 8. Volume over time — both phases
    ax = axes["volume_spinup"]
    for pt in PROFILE_TYPES:
        ds1 = p1[pt]
        ds2 = p2[pt]
        delx1 = float(ds1.attrs["delx"])
        delx2 = float(ds2.attrs["delx"])
        vol1 = (ds1["h"] * ds1["w"] * delx1).sum(dim="x") / 1e9
        vol2 = (ds2["h"] * ds2["w"] * delx2).sum(dim="x") / 1e9
        t_offset = float(ds1.coords["time"].max())
        ax.plot(ds1.coords["time"], vol1,
                color=COLORS[pt], linewidth=1.5, label=pt.replace("_", " ").title())
        ax.plot(ds2.coords["time"] + t_offset, vol2,
                color=COLORS[pt], linewidth=1.5, linestyle="--")
    ax.axvline(t_offset, color="k", linewidth=0.8, linestyle=":", alpha=0.6, label="Phase boundary")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Volume (km³)")
    ax.set_title("Glacier Volume: Phase 1 (solid) / Phase 2 (dashed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "width_profile_equilibrium_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Analysis plot saved to: {out_path}")

    # Print summary table
    print(f"\n{'Profile':<15} {'T0_p1 (°C)':>12} {'T0_p2 (°C)':>12} {'ΔT (°C)':>10} {'Vol_p1 (km³)':>14} {'Vol_p2 (km³)':>14}")
    print("-" * 80)
    for pt in PROFILE_TYPES:
        print(f"{pt:<15} {T0_p1[pt]:>12.3f} {T0_p2[pt]:>12.3f} {delta_T[pt]:>+10.3f} {vol_p1[pt]:>14.4f} {vol_p2[pt]:>14.4f}")


def main():
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = Path(__file__).resolve().parent / "output"
    plot(output_dir)


if __name__ == "__main__":
    main()
