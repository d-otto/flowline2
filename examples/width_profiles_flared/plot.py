#!/usr/bin/env python3
"""
Visualization for the width profiles flared example.

Reads output files from the sweep and produces the analysis plot.
Can be run standalone: uv run examples/width_profiles_flared/plot.py [output_dir]
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot(output_dir):
    output_dir = Path(output_dir)
    combined_results_path = output_dir / "combined_results.nc"

    if not combined_results_path.exists():
        raise FileNotFoundError(f"No combined_results.nc in {output_dir}")

    print("Creating width profiles flared analysis...")
    ds = xr.open_dataset(combined_results_path)
    delx = float(ds.attrs["delx"])

    ordered_profiles = ["upper_flared", "lower_flared", "lower_tapered", "upper_tapered"]
    colors = {
        "lower_flared":   "#1565C0",  # dark blue
        "upper_flared":   "#64B5F6",  # light blue
        "lower_tapered":  "#B71C1C",  # dark red
        "upper_tapered":  "#EF9A9A",  # light red/salmon
    }

    spinup_dir = output_dir / "spinup_profiles"
    spinup_P0 = {}
    V_eq = {}
    spinup_w = {}
    spinup_ela = {}
    spinup_ds = {}
    for profile_type in ordered_profiles:
        sp_path = spinup_dir / f"spinup_spinup_{profile_type}.nc"
        if sp_path.exists():
            sp = xr.open_dataset(sp_path)
            spinup_ds[profile_type] = sp
            spinup_P0[profile_type] = float(sp.attrs["forcing_P0"])
            sp_delx = float(sp.attrs["delx"])
            V_eq[profile_type] = float(
                (sp["h"].isel(time=-1) * sp["w"] * sp_delx).sum()
            )
            spinup_w[profile_type] = sp["w_geom_resampled"]
            if "ela" in sp:
                spinup_ela[profile_type] = float(sp["ela"].isel(time=-1))

    fig = plt.figure(figsize=(32, 18), layout="constrained")

    subplot_mosaic = [
        ["width_profiles",    "width_profiles",    "length_response",    "length_response_bar", "f_eq",           "f_eq",              "smb",               "smb"             ],
        ["volume_response",   "volume_response_bar","comparison",        "comparison_bar",      "flux",           "flux",              "terminus_mb",       "terminus_mb_bar" ],
        ["thickness_final",   "thickness_final",   "flux_pct_change",    "flux_pct_change",    "flux_pct_volume", "flux_pct_volume_bar","sensitivity",      "sensitivity"     ],
        ["plan_upper_flared", "plan_lower_flared",  "ela",               "ela_bar",            ".",               ".",                 ".",                 "."               ],
        ["plan_upper_tapered","plan_lower_tapered", "aar",               "aar_bar",            "cumsum_area",     "cumsum_area",       "cumsum_flux",       "cumsum_flux"     ],
    ]
    axes = fig.subplot_mosaic(subplot_mosaic)
    fig.suptitle("Width Profile Effects on Glacier Dynamics: Upper vs Lower Reach Flared/Tapered", fontsize=16)

    # 1. Width profile comparison
    for profile_type in ordered_profiles:
        if profile_type in ds.coords["run_id"].values:
            w_geom = ds["w_geom_resampled"].sel(run_id=profile_type)
            axes["width_profiles"].plot(
                ds.coords["x"] / 1000,
                w_geom,
                color=colors[profile_type],
                linewidth=2,
                alpha=0.7,
                label=profile_type.replace("_", " ").title(),
            )

    axes["width_profiles"].set_xlabel("Distance from head (km)")
    axes["width_profiles"].set_ylabel("Width (m)")
    axes["width_profiles"].set_title("Width Profile Comparison")
    axes["width_profiles"].legend(fontsize=7)
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
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["length_response"].set_xlabel("Time (years)")
        axes["length_response"].set_ylabel("Length (km)")
        axes["length_response"].set_title("Length Response to +1°C Warming")
        axes["length_response"].legend()
        axes["length_response"].grid(True, alpha=0.3)

    bar_width = 0.15
    n_profiles = len(ordered_profiles)
    bar_offsets = np.arange(n_profiles) - (n_profiles - 1) / 2

    def add_barplot(ax, data_initial, data_final, title, ylabel, data_spinup=None):
        if data_spinup is not None:
            x_groups = np.array([0.0, 1.0, 2.0])
            xlabels = ["Spinup", "Initial", "Final"]
        else:
            x_groups = np.array([0.0, 1.0])
            xlabels = ["Initial", "Final"]
        for i, profile_type in enumerate(ordered_profiles):
            if profile_type not in data_initial:
                continue
            offset = bar_offsets[i] * bar_width
            if data_spinup is not None:
                values = [
                    data_spinup.get(profile_type, float("nan")),
                    data_initial[profile_type],
                    data_final[profile_type],
                ]
            else:
                values = [data_initial[profile_type], data_final[profile_type]]
            ax.bar(
                x_groups + offset,
                values,
                width=bar_width,
                color=colors[profile_type],
                alpha=0.7,
            )
        ax.set_xticks(x_groups)
        ax.set_xticklabels(xlabels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    # 2b. Length barplot
    bar_length_initial = {}
    bar_length_final = {}
    if "edge" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                length_data = ds["edge"].sel(run_id=profile_type)
                bar_length_initial[profile_type] = float(length_data.isel(time=0)) / 1000
                bar_length_final[profile_type] = float(length_data.isel(time=-1)) / 1000
        add_barplot(axes["length_response_bar"], bar_length_initial, bar_length_final,
                    "Length: Initial vs Final", "Length (km)")

    # 3. Fractional equilibration: f_eq = V' / V'_eq
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
                alpha=0.7,
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
    spinup_P0_values = []

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
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

                if "edge" in ds.data_vars:
                    length_data = ds["edge"].sel(run_id=profile_type)
                    initial_lengths.append(float(length_data.isel(time=0)) / 1000)
                    final_lengths.append(float(length_data.isel(time=-1)) / 1000)
                    profile_names.append(profile_type.replace("_", " ").title())
                    spinup_P0_values.append(spinup_P0.get(profile_type))

        axes["volume_response"].set_xlabel("Time (years)")
        axes["volume_response"].set_ylabel("Volume (km³)")
        axes["volume_response"].set_title("Absolute Volume Response to +1°C Warming")
        axes["volume_response"].legend()
        axes["volume_response"].grid(True, alpha=0.3)

    elif "edge" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                length_data = ds["edge"].sel(run_id=profile_type)
                initial_lengths.append(float(length_data.isel(time=0)) / 1000)
                final_lengths.append(float(length_data.isel(time=-1)) / 1000)
                profile_names.append(profile_type.replace("_", " ").title())
                spinup_P0_values.append(spinup_P0.get(profile_type))

    # 4b. Volume barplot
    bar_volume_initial = {}
    bar_volume_final = {}
    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                vol = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x") / 1e9
                bar_volume_initial[profile_type] = float(vol.isel(time=0))
                bar_volume_final[profile_type] = float(vol.isel(time=-1))
        add_barplot(axes["volume_response_bar"], bar_volume_initial, bar_volume_final,
                    "Volume: Initial vs Final", "Volume (km³)")

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
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["comparison"].set_xlabel("Time (years)")
        axes["comparison"].set_ylabel("Area (km²)")
        axes["comparison"].set_title("Glacier Area Over Time")
        axes["comparison"].legend()
        axes["comparison"].grid(True, alpha=0.3)

    # 5b. Area barplot
    bar_area_initial = {}
    bar_area_final = {}
    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                h_profile = ds["h"].sel(run_id=profile_type)
                w_profile = ds["w"].sel(run_id=profile_type)
                area_km2 = w_profile.where(h_profile > 0, 0).sum(dim="x") * delx / 1e6
                bar_area_initial[profile_type] = float(area_km2.isel(time=0))
                bar_area_final[profile_type] = float(area_km2.isel(time=-1))
        add_barplot(axes["comparison_bar"], bar_area_initial, bar_area_final,
                    "Area: Initial vs Final", "Area (km²)")

    # 6. Ice flux profiles (initial and final)
    if "F" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                F_data = ds["F"].sel(run_id=profile_type)
                x_km = F_data.coords["x"] / 1000
                axes["flux"].plot(
                    x_km, F_data.isel(time=0),
                    color=colors[profile_type], linewidth=2, linestyle="--",
                )
                axes["flux"].plot(
                    x_km, F_data.isel(time=-1),
                    color=colors[profile_type], linewidth=2, label=profile_type.replace("_", " ").title(),
                )

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
                axes["thickness_final"].plot(
                    x_km, h_data.isel(time=0),
                    color=colors[profile_type], linewidth=2, linestyle="--",
                )
                axes["thickness_final"].plot(
                    x_km, h_data.isel(time=-1),
                    color=colors[profile_type], linewidth=2, label=profile_type.replace("_", " ").title(),
                )

        axes["thickness_final"].set_xlabel("Distance from head (km)")
        axes["thickness_final"].set_ylabel("Ice thickness (m)")
        axes["thickness_final"].set_title("Ice Thickness Profiles (solid=final, dashed=initial)")
        axes["thickness_final"].legend()
        axes["thickness_final"].grid(True, alpha=0.3)

    # 8. Sensitivity analysis: spinup P0 and length change
    valid_P0 = [p is not None for p in spinup_P0_values]
    if profile_names and all(valid_P0) and initial_lengths and final_lengths:
        length_change = np.array(final_lengths) - np.array(initial_lengths)

        ax_p0 = axes["sensitivity"]
        ax_change = ax_p0.twinx()
        x_pos = np.arange(len(profile_names))

        line1 = ax_p0.plot(
            x_pos, spinup_P0_values,
            "o-", color="orange", linewidth=2, markersize=8, label="Spinup P0 (m/yr)",
        )
        line2 = ax_change.plot(
            x_pos, length_change,
            "s-", color="purple", linewidth=2, markersize=8, label="Length Change (km)",
        )

        ax_p0.set_xlabel("Width Profile Type")
        ax_p0.set_ylabel("Spinup P0 (m/yr)", color="orange")
        ax_change.set_ylabel("Length Change (km)", color="purple")
        ax_p0.set_title("Width Profile Sensitivity Analysis")
        ax_p0.set_xticks(x_pos)
        ax_p0.set_xticklabels(profile_names, rotation=45, ha="right")
        ax_p0.grid(True, alpha=0.3)
        ax_change.invert_yaxis()
        lines = line1 + line2
        ax_p0.legend(lines, [line.get_label() for line in lines], loc="upper left")

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
                    f_flux.coords["time"],
                    f_flux,
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["flux_pct_change"].axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        axes["flux_pct_change"].axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["flux_pct_change"].set_xlabel("Time (years)")
        axes["flux_pct_change"].set_ylabel("f_flux = (F/V - (F/V)₀) / ((F/V)_eq - (F/V)₀)")
        axes["flux_pct_change"].set_title("Fractional Equilibration of Flux/Volume Ratio")
        axes["flux_pct_change"].legend()
        axes["flux_pct_change"].grid(True, alpha=0.3)

    # 10. Total flux as a percentage of volume
    if "F" in ds.data_vars and "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                total_flux = ds["F"].sel(run_id=profile_type).sum(dim="x")
                volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
                flux_pct_volume = total_flux / volume * 100
                axes["flux_pct_volume"].plot(
                    flux_pct_volume.coords["time"],
                    flux_pct_volume,
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["flux_pct_volume"].set_xlabel("Time (years)")
        axes["flux_pct_volume"].set_ylabel("Total flux / volume (%)")
        axes["flux_pct_volume"].set_title("Total Ice Flux as % of Volume")
        axes["flux_pct_volume"].legend()
        axes["flux_pct_volume"].grid(True, alpha=0.3)

    # 10b. Flux % of volume barplot
    bar_fv_initial = {}
    bar_fv_final = {}
    if "F" in ds.data_vars and "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                total_flux = ds["F"].sel(run_id=profile_type).sum(dim="x")
                volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
                flux_pct = total_flux / volume * 100
                bar_fv_initial[profile_type] = float(flux_pct.isel(time=0))
                bar_fv_final[profile_type] = float(flux_pct.isel(time=-1))
        add_barplot(axes["flux_pct_volume_bar"], bar_fv_initial, bar_fv_final,
                    "Flux % Volume: Initial vs Final", "Flux / Volume (%)")

    # 11. Plan view of each glacier
    plan_axes_map = {
        "upper_flared":   "plan_upper_flared",
        "lower_flared":   "plan_lower_flared",
        "lower_tapered":  "plan_lower_tapered",
        "upper_tapered":  "plan_upper_tapered",
    }
    if "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            ax = axes[plan_axes_map[profile_type]]
            color = colors[profile_type]
            if profile_type in ds.coords["run_id"].values:
                h_data = ds["h"].sel(run_id=profile_type)
                x_vals = h_data.coords["x"].values
                w_geom = ds["w_geom_resampled"].sel(run_id=profile_type).values

                for time_idx, linestyle, time_label in [
                    (0, "--", "Initial"),
                    (-1, "-", "Final"),
                ]:
                    h = h_data.isel(time=time_idx).values
                    half_w = np.where(h > 0, w_geom / 2, 0)
                    ax.plot(half_w,  x_vals, color=color, linestyle=linestyle, linewidth=2, label=time_label)
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
            label = profile_type.replace("_", " ").title()
            ax.set_title(f"Plan View: {label}", fontsize=9)
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)

    # 12. Terminus mass balance over time
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
                    terminus_mb.coords["time"],
                    terminus_mb,
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["terminus_mb"].set_xlabel("Time (years)")
        axes["terminus_mb"].set_ylabel("Mass balance (m/yr)")
        axes["terminus_mb"].set_title("Terminus Mass Balance Over Time")
        axes["terminus_mb"].legend()
        axes["terminus_mb"].grid(True, alpha=0.3)

    # 12b. Terminus mass balance barplot
    bar_tmb_spinup = {}
    bar_tmb_initial = {}
    bar_tmb_final = {}
    if "b_profile" in ds.data_vars and "edge_idx" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in spinup_ds:
                sp = spinup_ds[profile_type]
                bar_tmb_spinup[profile_type] = float(
                    sp["b_profile"].isel(time=-1).isel(x=int(sp["edge_idx"].isel(time=-1)))
                )
            if profile_type in ds.coords["run_id"].values:
                terminus_mb = ds["b_profile"].sel(run_id=profile_type).isel(
                    x=ds["edge_idx"].sel(run_id=profile_type)
                )
                bar_tmb_initial[profile_type] = float(terminus_mb.isel(time=0))
                bar_tmb_final[profile_type] = float(terminus_mb.isel(time=-1))
        add_barplot(axes["terminus_mb_bar"], bar_tmb_initial, bar_tmb_final,
                    "Terminus MB: Spinup / Initial / Final", "Mass balance (m/yr)",
                    data_spinup=bar_tmb_spinup)

    # 13. Specific mass balance over time
    if "total_mass_balance" in ds.data_vars and "area" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in ds.coords["run_id"].values:
                smb = ds["total_mass_balance"].sel(run_id=profile_type) / ds["area"].sel(run_id=profile_type)
                axes["smb"].plot(
                    smb.coords["time"],
                    smb,
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["smb"].axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["smb"].set_xlabel("Time (years)")
        axes["smb"].set_ylabel("Specific mass balance (m/yr)")
        axes["smb"].set_title("Specific Mass Balance Over Time")
        axes["smb"].legend()
        axes["smb"].grid(True, alpha=0.3)

    # 14. ELA: spinup endpoint as scatter, then ΔELA timeseries relative to spinup, plus barplot
    bar_ela_spinup = {}
    bar_ela_initial = {}
    bar_ela_final = {}
    if "ela" in ds.data_vars:
        t_spinup_dummy = float(ds.coords["time"].isel(time=0)) - 1  # just before t=0 for scatter x
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            ela = ds["ela"].sel(run_id=profile_type)
            ela_spinup_val = spinup_ela.get(profile_type, float(ela.isel(time=0)))
            label = profile_type.replace("_", " ").title()
            # Spinup ELA as a point just before experiment start
            axes["ela"].scatter(
                [t_spinup_dummy],
                [ela_spinup_val],
                color=colors[profile_type],
                s=50,
                zorder=5,
            )
            axes["ela"].plot(
                ela.coords["time"],
                ela,
                color=colors[profile_type],
                linewidth=2,
                alpha=0.7,
                label=label,
            )
            bar_ela_spinup[profile_type] = ela_spinup_val
            bar_ela_initial[profile_type] = float(ela.isel(time=0))
            bar_ela_final[profile_type] = float(ela.isel(time=-1))

        axes["ela"].set_xlabel("Time (years)")
        axes["ela"].set_ylabel("ELA (m)")
        axes["ela"].set_title("ELA over time")
        axes["ela"].legend(fontsize=7)
        axes["ela"].grid(True, alpha=0.3)

        add_barplot(axes["ela_bar"], bar_ela_initial, bar_ela_final,
                    "ELA: Spinup / Initial / Final", "ELA (m)",
                    data_spinup=bar_ela_spinup)

    # 16. Accumulation area ratio (AAR) over time
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
                    aar.coords["time"],
                    aar,
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

        axes["aar"].set_xlabel("Time (years)")
        axes["aar"].set_ylabel("AAR")
        axes["aar"].set_title("Accumulation Area Ratio Over Time")
        axes["aar"].legend()
        axes["aar"].grid(True, alpha=0.3)

    # 16b. AAR barplot
    bar_aar_spinup = {}
    bar_aar_initial = {}
    bar_aar_final = {}
    if "b_profile" in ds.data_vars and "w" in ds.data_vars and "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type in spinup_ds:
                sp = spinup_ds[profile_type]
                sp_delx = float(sp.attrs["delx"])
                sp_b = sp["b_profile"].isel(time=-1)
                sp_w = sp["w"]
                sp_h = sp["h"].isel(time=-1)
                sp_accum = float(sp_w.where(sp_b > 0, 0).sum() * sp_delx)
                sp_total = float(sp_w.where(sp_h > 0, 0).sum() * sp_delx)
                bar_aar_spinup[profile_type] = sp_accum / sp_total
            if profile_type in ds.coords["run_id"].values:
                b = ds["b_profile"].sel(run_id=profile_type)
                w = ds["w"].sel(run_id=profile_type)
                h = ds["h"].sel(run_id=profile_type)
                accum = w.where(b > 0, 0).sum(dim="x") * delx
                total = w.where(h > 0, 0).sum(dim="x") * delx
                aar_series = accum / total
                bar_aar_initial[profile_type] = float(aar_series.isel(time=0))
                bar_aar_final[profile_type] = float(aar_series.isel(time=-1))
        add_barplot(axes["aar_bar"], bar_aar_initial, bar_aar_final,
                    "AAR: Spinup / Initial / Final", "AAR",
                    data_spinup=bar_aar_spinup)

    # 17. Cumulative area along x-axis (spinup final timestep)
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
                color=colors[profile_type],
                linewidth=2,
                alpha=0.7,
                label=profile_type.replace("_", " ").title(),
            )

    axes["cumsum_area"].set_xlabel("Distance from head (km)")
    axes["cumsum_area"].set_ylabel("Cumulative area (km²)")
    axes["cumsum_area"].set_title("Cumulative Glacier Area Along Flowline (spinup final state)")
    axes["cumsum_area"].legend()
    axes["cumsum_area"].grid(True, alpha=0.3)

    # 18. Cumulative ice flux along x-axis (spinup final timestep)
    for profile_type in ordered_profiles:
        if profile_type in spinup_ds:
            sp = spinup_ds[profile_type]
            if "F" in sp:
                x_sp = sp.coords["x"].values
                F_final = sp["F"].isel(time=-1).values
                axes["cumsum_flux"].plot(
                    x_sp / 1000,
                    np.cumsum(F_final),
                    color=colors[profile_type],
                    linewidth=2,
                alpha=0.7,
                    label=profile_type.replace("_", " ").title(),
                )

    axes["cumsum_flux"].set_xlabel("Distance from head (km)")
    axes["cumsum_flux"].set_ylabel("Cumulative ice flux (m³/yr)")
    axes["cumsum_flux"].set_title("Cumulative Ice Flux Along Flowline (spinup final state)")
    axes["cumsum_flux"].legend()
    axes["cumsum_flux"].grid(True, alpha=0.3)

    plot_path = output_dir / "width_profiles_flared_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis plot saved to: {plot_path}")

    if spinup_P0_values and initial_lengths and final_lengths:
        print("\nWidth Profiles Flared Analysis Summary:")
        print("  Target volume: 1e9 m³ (1 km³, achieved through P0 optimization)")
        print("\n  Spinup P0 values (m/yr):")
        for i, name in enumerate(profile_names):
            if spinup_P0_values[i] is not None:
                print(f"    {name}: {spinup_P0_values[i]:.3f} m/yr")
        print("\n  Initial lengths (post-spinup):")
        for i, name in enumerate(profile_names):
            print(f"    {name}: {initial_lengths[i]:.1f} km")
        print("\n  Final lengths (after +1°C):")
        for i, name in enumerate(profile_names):
            print(f"    {name}: {final_lengths[i]:.1f} km")
        print("\n  Length change from +1°C warming:")
        for i, name in enumerate(profile_names):
            change = final_lengths[i] - initial_lengths[i]
            print(f"    {name}: {change:+.1f} km")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "output")
    plot(output_dir)
