#!/usr/bin/env python3
"""
Visualization for the width harmonic variations example.

Reads output files from the sweep and produces the analysis plot.
Can be run standalone: uv run examples/width_harmonic_variations/plot.py [output_dir]
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_exponential(t, y):
    """Fit y = 1 - exp(-t/tau) to fractional equilibration. Returns tau or nan."""
    def model(t, tau):
        return 1 - np.exp(-t / tau)
    try:
        popt, _ = curve_fit(model, t, y, p0=[50.0], bounds=(1e-3, 1e6))
        return float(popt[0])
    except Exception:
        return np.nan


def plot(output_dir):
    output_dir = Path(output_dir)
    combined_results_path = output_dir / "combined_results.nc"

    if not combined_results_path.exists():
        raise FileNotFoundError(f"No combined_results.nc in {output_dir}")

    print("Creating width harmonic analysis...")
    ds = xr.open_dataset(combined_results_path)
    delx = float(ds.attrs["delx"])

    ordered_profiles = ["hourglass", "oval", "neutral"]
    colors = {"hourglass": "teal", "oval": "purple", "neutral": "green"}

    spinup_dir = output_dir / "spinup_profiles"
    spinup_T0 = {}
    V_eq = {}
    spinup_w = {}
    spinup_ela = {}
    spinup_ds = {}
    for profile_type in ordered_profiles:
        sp_path = spinup_dir / f"spinup_spinup_{profile_type}.nc"
        if sp_path.exists():
            sp = xr.open_dataset(sp_path)
            spinup_ds[profile_type] = sp
            spinup_T0[profile_type] = float(sp.attrs["forcing_T0"])
            sp_delx = float(sp.attrs["delx"])
            V_eq[profile_type] = float(
                (sp["h"].isel(time=-1) * sp["w"] * sp_delx).sum()
            )
            spinup_w[profile_type] = sp["w_geom_resampled"]
            if "ela" in sp:
                spinup_ela[profile_type] = float(sp["ela"].isel(time=-1))

    fig = plt.figure(figsize=(32, 21), layout="constrained")

    subplot_mosaic = [
        ["width_profiles", "width_profiles", "length_response", "length_response", "f_eq",           "f_eq",            "smb",         "smb"        ],
        ["volume_response", "volume_response", "comparison",    "comparison",       "flux",            "flux",            "terminus_mb", "terminus_mb"],
        ["thickness_final", "thickness_final", "flux_pct_change","flux_pct_change", "flux_pct_volume", "flux_pct_volume", "sensitivity", "sensitivity"],
        ["plan_hourglass",  "plan_hourglass",  "plan_oval",     "plan_oval",        "plan_neutral",    "plan_neutral",    "ela",         "ela"        ],
        ["cumsum_area",     "cumsum_area",     "tau",           "tau",              "ela_t0",          "ela_t0",          "aar",         "aar"        ],
        ["beta_ablation",   "beta_ablation",   "beta_global",   "beta_global",      "beta_length",     "beta_length",     ".",           "."          ],
    ]
    axes = fig.subplot_mosaic(subplot_mosaic)
    fig.suptitle("Width Shape Effects on Glacier Dynamics: Harmonic Profiles (Hourglass vs Oval)", fontsize=16)

    # 1. Width profile comparison
    for profile_type in ordered_profiles:
        if profile_type in ds.coords["run_id"].values:
            w_geom = ds["w_geom_resampled"].sel(run_id=profile_type)
            axes["width_profiles"].plot(
                ds.coords["x"] / 1000,
                w_geom,
                color=colors[profile_type],
                linewidth=2,
                label=profile_type.replace("_", " ").title(),
            )

    axes["width_profiles"].set_xlabel("Distance from head (km)")
    axes["width_profiles"].set_ylabel("Width (m)")
    axes["width_profiles"].set_title("Harmonic Width Profiles")
    axes["width_profiles"].legend()
    axes["width_profiles"].grid(True, alpha=0.3)

    # 2. Length response trajectories
    if "edge" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                length_km = ds["edge"].sel(run_id=profile_type) / 1000
                axes["length_response"].plot(
                    length_km.coords["time"],
                    length_km,
                    color=colors[profile_type],
                    linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["length_response"].set_xlabel("Time (years)")
        axes["length_response"].set_ylabel("Length (km)")
        axes["length_response"].set_title("Length Response to +2°C Warming")
        axes["length_response"].legend()
        axes["length_response"].grid(True, alpha=0.3)

    # 3. Fractional equilibration
    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
                V_0 = float(volume.isel(time=0))
                V_eq_response = float(volume.isel(time=-1))
                f_eq = (volume - V_0) / (V_eq_response - V_0)
                axes["f_eq"].plot(
                    f_eq.coords["time"],
                    f_eq,
                    color=colors[profile_type],
                    linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["f_eq"].axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        axes["f_eq"].axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["f_eq"].set_xlabel("Time (years)")
        axes["f_eq"].set_ylabel("f_eq = (V - V₀) / (V_eq - V₀)")
        axes["f_eq"].set_title("Fractional Equilibration")
        axes["f_eq"].legend()
        axes["f_eq"].grid(True, alpha=0.3)

    # 4. Absolute volume over time
    profile_names = []
    initial_lengths = []
    final_lengths = []
    spinup_T0_values = []

    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                volume_km3 = (
                    ds["h"].sel(run_id=profile_type)
                    * ds["w"].sel(run_id=profile_type)
                    * delx
                ).sum(dim="x") / 1e9
                axes["volume_response"].plot(
                    volume_km3.coords["time"],
                    volume_km3,
                    color=colors[profile_type],
                    linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

                if "edge" in ds.data_vars:
                    length_data = ds["edge"].sel(run_id=profile_type)
                    initial_lengths.append(float(length_data.isel(time=0)) / 1000)
                    final_lengths.append(float(length_data.isel(time=-1)) / 1000)
                    profile_names.append(profile_type.replace("_", " ").title())
                    spinup_T0_values.append(spinup_T0.get(profile_type))

        axes["volume_response"].set_xlabel("Time (years)")
        axes["volume_response"].set_ylabel("Volume (km³)")
        axes["volume_response"].set_title("Absolute Volume Response to +2°C Warming")
        axes["volume_response"].legend()
        axes["volume_response"].grid(True, alpha=0.3)

    elif "edge" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                length_data = ds["edge"].sel(run_id=profile_type)
                initial_lengths.append(float(length_data.isel(time=0)) / 1000)
                final_lengths.append(float(length_data.isel(time=-1)) / 1000)
                profile_names.append(profile_type.replace("_", " ").title())
                spinup_T0_values.append(spinup_T0.get(profile_type))

    # 5. Glacier area over time
    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                h_profile = ds["h"].sel(run_id=profile_type)
                w_profile = ds["w"].sel(run_id=profile_type)
                area_km2 = w_profile.where(h_profile > 0, 0).sum(dim="x") * delx / 1e6
                axes["comparison"].plot(
                    area_km2.coords["time"],
                    area_km2,
                    color=colors[profile_type],
                    linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["comparison"].set_xlabel("Time (years)")
        axes["comparison"].set_ylabel("Area (km²)")
        axes["comparison"].set_title("Glacier Area Over Time")
        axes["comparison"].legend()
        axes["comparison"].grid(True, alpha=0.3)

    # 6. Ice flux profiles
    if "F" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                F_data = ds["F"].sel(run_id=profile_type)
                x_km = F_data.coords["x"] / 1000
                label = profile_type.replace("_", " ").title()
                axes["flux"].plot(x_km, F_data.isel(time=0), color=colors[profile_type], linewidth=2, linestyle="--")
                axes["flux"].plot(x_km, F_data.isel(time=-1), color=colors[profile_type], linewidth=2, label=label)

        axes["flux"].set_xlabel("Distance from head (km)")
        axes["flux"].set_ylabel("Ice flux (m³/yr)")
        axes["flux"].set_title("Ice Flux Profile (solid=final, dashed=initial)")
        axes["flux"].legend()
        axes["flux"].grid(True, alpha=0.3)

    # 7. Final and initial thickness profiles
    if "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                h_data = ds["h"].sel(run_id=profile_type)
                x_km = h_data.coords["x"] / 1000
                label = profile_type.replace("_", " ").title()
                axes["thickness_final"].plot(x_km, h_data.isel(time=0), color=colors[profile_type], linewidth=2, linestyle="--")
                axes["thickness_final"].plot(x_km, h_data.isel(time=-1), color=colors[profile_type], linewidth=2, label=label)

        axes["thickness_final"].set_xlabel("Distance from head (km)")
        axes["thickness_final"].set_ylabel("Ice thickness (m)")
        axes["thickness_final"].set_title("Ice Thickness Profiles (solid=final, dashed=initial)")
        axes["thickness_final"].legend()
        axes["thickness_final"].grid(True, alpha=0.3)

    # 8. Sensitivity analysis
    valid_T0 = [t is not None for t in spinup_T0_values]
    if profile_names and all(valid_T0) and initial_lengths and final_lengths:
        length_change = np.array(final_lengths) - np.array(initial_lengths)

        ax_t0 = axes["sensitivity"]
        ax_change = ax_t0.twinx()
        x_pos = np.arange(len(profile_names))

        line1 = ax_t0.plot(x_pos, spinup_T0_values, "o-", color="orange", linewidth=2, markersize=8, label="Spinup T0 (°C)")
        line2 = ax_change.plot(x_pos, length_change, "s-", color="purple", linewidth=2, markersize=8, label="Length Change (km)")

        ax_t0.set_xlabel("Width Shape")
        ax_t0.set_ylabel("Spinup T0 (°C)", color="orange")
        ax_change.set_ylabel("Length Change (km)", color="purple")
        ax_t0.set_title("Width Shape Sensitivity Analysis")
        ax_t0.set_xticks(x_pos)
        ax_t0.set_xticklabels(profile_names, rotation=45, ha="right")
        ax_t0.grid(True, alpha=0.3)
        ax_change.invert_yaxis()
        lines = line1 + line2
        ax_t0.legend(lines, [line.get_label() for line in lines], loc="upper left")

    # 9. Fractional equilibration of flux/volume ratio
    if "F" in ds.data_vars and "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                total_flux = ds["F"].sel(run_id=profile_type).sum(dim="x")
                volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
                flux_per_vol = total_flux / volume
                fv_0 = float(flux_per_vol.isel(time=0))
                fv_end = float(flux_per_vol.isel(time=-1))
                f_flux = (flux_per_vol - fv_0) / (fv_end - fv_0)
                axes["flux_pct_change"].plot(
                    f_flux.coords["time"], f_flux,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["flux_pct_change"].axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        axes["flux_pct_change"].axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["flux_pct_change"].set_xlabel("Time (years)")
        axes["flux_pct_change"].set_ylabel("f_flux = (F/V - (F/V)₀) / ((F/V)_eq - (F/V)₀)")
        axes["flux_pct_change"].set_title("Fractional Equilibration of Flux/Volume Ratio")
        axes["flux_pct_change"].legend()
        axes["flux_pct_change"].grid(True, alpha=0.3)

    # 10. Total flux as percentage of volume
    if "F" in ds.data_vars and "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                total_flux = ds["F"].sel(run_id=profile_type).sum(dim="x")
                volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
                flux_pct_volume = total_flux / volume * 100
                axes["flux_pct_volume"].plot(
                    flux_pct_volume.coords["time"], flux_pct_volume,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["flux_pct_volume"].set_xlabel("Time (years)")
        axes["flux_pct_volume"].set_ylabel("Total flux / volume (%)")
        axes["flux_pct_volume"].set_title("Total Ice Flux as % of Volume")
        axes["flux_pct_volume"].legend()
        axes["flux_pct_volume"].grid(True, alpha=0.3)

    # 11. Plan view of each glacier
    plan_axes_map = {"hourglass": "plan_hourglass", "oval": "plan_oval", "neutral": "plan_neutral"}
    if "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            ax = axes[plan_axes_map[profile_type]]
            color = colors[profile_type]
            if profile_type in ds.coords["run_id"].values:
                h_data = ds["h"].sel(run_id=profile_type)
                x_vals = h_data.coords["x"].values
                w_geom = ds["w_geom_resampled"].sel(run_id=profile_type).values

                for time_idx, linestyle, time_label in [(0, "--", "Initial"), (-1, "-", "Final")]:
                    h = h_data.isel(time=time_idx).values
                    half_w = np.where(h > 0, w_geom / 2, 0)
                    ax.plot(half_w, x_vals, color=color, linestyle=linestyle, linewidth=2, label=time_label)
                    ax.plot(-half_w, x_vals, color=color, linestyle=linestyle, linewidth=2)
                    ax.fill_betweenx(x_vals, -half_w, half_w, alpha=0.08, color=color)

                if "zb" in ds.data_vars:
                    zb = ds["zb"].sel(run_id=profile_type).values
                    max_w = w_geom.max()
                    ela_elevs = [
                        (spinup_ela.get(profile_type), "--", "ELA (initial)"),
                        (float(ds["ela"].sel(run_id=profile_type).isel(time=-1)) if "ela" in ds else None, "-", "ELA (final)"),
                    ]
                    for ela_elev, linestyle, ela_label in ela_elevs:
                        if ela_elev is None:
                            continue
                        ela_x = x_vals[np.searchsorted(-zb, -ela_elev)]
                        ax.plot([-max_w, max_w], [ela_x, ela_x], color="black", linestyle=linestyle, linewidth=1.5, label=ela_label)

            ax.set_xlabel("Width (m)")
            ax.set_ylabel("Distance from head (m)")
            ax.set_title(f"Plan View: {profile_type.replace('_', ' ').title()}")
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)

    # 12. Terminus mass balance
    if "b_profile" in ds.data_vars and "edge_idx" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                terminus_mb = ds["b_profile"].sel(run_id=profile_type).isel(
                    x=ds["edge_idx"].sel(run_id=profile_type)
                )
                if profile_type in spinup_ds:
                    sp = spinup_ds[profile_type]
                    sp_val = float(sp["b_profile"].isel(time=-1).isel(x=int(sp["edge_idx"].isel(time=-1))))
                    axes["terminus_mb"].scatter([0], [sp_val], color=colors[profile_type], s=40, zorder=5)
                axes["terminus_mb"].plot(
                    terminus_mb.coords["time"], terminus_mb,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["terminus_mb"].axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["terminus_mb"].set_xlabel("Time (years)")
        axes["terminus_mb"].set_ylabel("Mass balance (m/yr)")
        axes["terminus_mb"].set_title("Terminus Mass Balance Over Time")
        axes["terminus_mb"].legend()
        axes["terminus_mb"].grid(True, alpha=0.3)

    # 13. Specific mass balance
    if "total_mass_balance" in ds.data_vars and "area" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                smb = ds["total_mass_balance"].sel(run_id=profile_type) / ds["area"].sel(run_id=profile_type)
                axes["smb"].plot(
                    smb.coords["time"], smb,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["smb"].axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["smb"].set_xlabel("Time (years)")
        axes["smb"].set_ylabel("Specific mass balance (m/yr)")
        axes["smb"].set_title("Specific Mass Balance Over Time")
        axes["smb"].legend()
        axes["smb"].grid(True, alpha=0.3)

    # 14. ELA change (relative to spinup)
    if "ela" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                ela = ds["ela"].sel(run_id=profile_type)
                ela_0 = spinup_ela.get(profile_type, float(ela.isel(time=0)))
                ela_change = ela - ela_0
                axes["ela"].plot(
                    ela_change.coords["time"], ela_change,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["ela"].set_xlabel("Time (years)")
        axes["ela"].set_ylabel("ΔELA (m)")
        axes["ela"].set_title("ELA Change (relative to spinup)")
        axes["ela"].legend()
        axes["ela"].grid(True, alpha=0.3)

    # 15. ELA change (relative to experiment t=0)
    if "ela" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                ela = ds["ela"].sel(run_id=profile_type)
                ela_change = ela - float(ela.isel(time=0))
                axes["ela_t0"].plot(
                    ela_change.coords["time"], ela_change,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["ela_t0"].set_xlabel("Time (years)")
        axes["ela_t0"].set_ylabel("ΔELA (m)")
        axes["ela_t0"].set_title("ELA Change (relative to t=0)")
        axes["ela_t0"].legend()
        axes["ela_t0"].grid(True, alpha=0.3)

    # 16. Accumulation area ratio
    if "b_profile" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                b = ds["b_profile"].sel(run_id=profile_type)
                w = ds["w"].sel(run_id=profile_type)
                accum_area = w.where(b > 0, 0).sum(dim="x") * delx
                total_area = w.where(ds["h"].sel(run_id=profile_type) > 0, 0).sum(dim="x") * delx
                aar = accum_area / total_area
                if profile_type in spinup_ds:
                    sp = spinup_ds[profile_type]
                    sp_b = sp["b_profile"].isel(time=-1)
                    sp_w = sp["w"]
                    sp_h = sp["h"].isel(time=-1)
                    sp_delx = float(sp.attrs["delx"])
                    sp_accum = float(sp_w.where(sp_b > 0, 0).sum() * sp_delx)
                    sp_total = float(sp_w.where(sp_h > 0, 0).sum() * sp_delx)
                    axes["aar"].scatter([0], [sp_accum / sp_total], color=colors[profile_type], s=40, zorder=5)
                axes["aar"].plot(
                    aar.coords["time"], aar,
                    color=colors[profile_type], linewidth=2,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["aar"].set_xlabel("Time (years)")
        axes["aar"].set_ylabel("AAR")
        axes["aar"].set_title("Accumulation Area Ratio Over Time")
        axes["aar"].legend()
        axes["aar"].grid(True, alpha=0.3)

    # 17. Cumulative area along flowline (spinup final state)
    for profile_type in ordered_profiles:
        if profile_type in spinup_ds:
            sp = spinup_ds[profile_type]
            sp_delx = float(sp.attrs["delx"])
            h_final = sp["h"].isel(time=-1).values
            w = sp["w"].values
            x_sp = sp.coords["x"].values
            cell_area = np.where(h_final > 0, w * sp_delx, 0)
            axes["cumsum_area"].plot(
                x_sp / 1000,
                np.cumsum(cell_area) / 1e6,
                color=colors[profile_type], linewidth=2,
                label=profile_type.replace("_", " ").title(),
            )

    axes["cumsum_area"].set_xlabel("Distance from head (km)")
    axes["cumsum_area"].set_ylabel("Cumulative area (km²)")
    axes["cumsum_area"].set_title("Cumulative Glacier Area Along Flowline (spinup final state)")
    axes["cumsum_area"].legend()
    axes["cumsum_area"].grid(True, alpha=0.3)

    # 18. Tau: response timescale comparison
    # Theoretical tau = -H_abl / b_term from spinup state (Jóhannesson et al. 1989)
    # Also fits an exponential to the fractional volume equilibration.
    tau_theory = {}
    tau_fitted = {}
    for profile_type in ordered_profiles:
        if profile_type not in spinup_ds:
            continue
        sp = spinup_ds[profile_type]
        h_sp = sp["h"].isel(time=-1)
        b_sp = sp["b_profile"].isel(time=-1)
        edge_idx_sp = int(sp["edge_idx"].isel(time=-1))
        ice_mask = h_sp > 0
        abl_mask = ice_mask & (b_sp < 0)
        b_term = float(b_sp.isel(x=edge_idx_sp))
        h_abl_mean = float(h_sp.where(abl_mask).mean())
        if abs(b_term) > 1e-10:
            tau_theory[profile_type] = -h_abl_mean / b_term

        if "h" in ds.data_vars and "w" in ds.data_vars and profile_type in ds.coords["run_id"].values:
            volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
            vol_arr = volume.values
            t_arr = volume.coords["time"].values.astype(float)
            vol_range = vol_arr[-1] - vol_arr[0]
            if abs(vol_range) > 1e-10:
                f_vol = (vol_arr - vol_arr[0]) / vol_range
                tau_fitted[profile_type] = fit_exponential(t_arr, f_vol)

    if tau_theory or tau_fitted:
        ax_tau = axes["tau"]
        profile_labels = [p.replace("_", " ").title() for p in ordered_profiles]
        x_pos = np.arange(len(ordered_profiles))

        theory_vals = [tau_theory.get(p, np.nan) for p in ordered_profiles]
        fitted_vals = [tau_fitted.get(p, np.nan) for p in ordered_profiles]

        ax_tau.plot(x_pos, theory_vals, "o-", color="steelblue", linewidth=2, markersize=8, label="Theory: -H_abl / b_term")
        ax_tau.plot(x_pos, fitted_vals, "s--", color="firebrick", linewidth=2, markersize=8, label="Fit: exp(-t/tau) to f_vol")

        ax_tau.set_xticks(x_pos)
        ax_tau.set_xticklabels(profile_labels, rotation=45, ha="right")
        ax_tau.set_ylabel("Response timescale tau (years)")
        ax_tau.set_title("Response Timescale by Width Shape")
        ax_tau.legend()
        ax_tau.grid(True, alpha=0.3)

    # 19. Beta factors

    if "h" in ds.data_vars and "w" in ds.data_vars and "b_profile" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            h = ds["h"].sel(run_id=profile_type)
            w_ice = ds["w"].sel(run_id=profile_type)
            b = ds["b_profile"].sel(run_id=profile_type)

            total_area = w_ice.where(h > 0, 0).sum(dim="x") * delx

            ablation_mask = b < 0
            ice_mask = h > 0
            combined_mask = ablation_mask & ice_mask

            w_abl = w_ice.where(combined_mask).mean(dim="x")
            h_abl = h.where(combined_mask).mean(dim="x")
            beta_ablation = total_area / (w_abl * h_abl)

            w_mean = w_ice.where(ice_mask).mean(dim="x")
            h_mean = h.where(ice_mask).mean(dim="x")
            beta_global = total_area / (w_mean * h_mean)

            length = ds["edge"].sel(run_id=profile_type)
            beta_length = length / h_abl

            label = profile_type.replace("_", " ").title()
            color = colors[profile_type]
            t = h.coords["time"]
            for ax_key, beta_da in [
                ("beta_ablation", beta_ablation),
                ("beta_global", beta_global),
                ("beta_length", beta_length),
            ]:
                axes[ax_key].plot(t, beta_da, color=color, linewidth=2, label=label)

        axes["beta_ablation"].set_title("Beta: area / (w_abl * h_abl)")
        axes["beta_ablation"].set_xlabel("Time (years)")
        axes["beta_ablation"].set_ylabel("Beta (ablation zone mean)")
        axes["beta_ablation"].legend()
        axes["beta_ablation"].grid(True, alpha=0.3)

        axes["beta_global"].set_title("Beta: area / (w_mean * h_mean)")
        axes["beta_global"].set_xlabel("Time (years)")
        axes["beta_global"].set_ylabel("Beta (whole-glacier mean)")
        axes["beta_global"].legend()
        axes["beta_global"].grid(True, alpha=0.3)

        axes["beta_length"].set_title("Beta: length / h_below_ela")
        axes["beta_length"].set_xlabel("Time (years)")
        axes["beta_length"].set_ylabel("Beta (length / mean h below ELA)")
        axes["beta_length"].legend()
        axes["beta_length"].grid(True, alpha=0.3)

    plot_path = output_dir / "width_harmonic_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis plot saved to: {plot_path}")

    if spinup_T0_values and initial_lengths and final_lengths:
        print("\nWidth Harmonic Analysis Summary:")
        print("  All profiles: offset=1250m, R=750m -> min=500m, max=2000m, mean=1250m")
        print("\n  Spinup T0 values (pre-perturbation):")
        for i, name in enumerate(profile_names):
            if spinup_T0_values[i] is not None:
                print(f"    {name}: {spinup_T0_values[i]:.3f}°C")
        print("\n  Initial lengths (post-spinup):")
        for i, name in enumerate(profile_names):
            print(f"    {name}: {initial_lengths[i]:.1f}km")
        print("\n  Final lengths (after +2°C):")
        for i, name in enumerate(profile_names):
            print(f"    {name}: {final_lengths[i]:.1f}km")
        print("\n  Length change from +2°C warming:")
        for i, name in enumerate(profile_names):
            change = final_lengths[i] - initial_lengths[i]
            print(f"    {name}: {change:+.1f}km")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "output")
    plot(output_dir)
