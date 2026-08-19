#!/usr/bin/env python3
"""
Linear model comparison plot for width shape variations.

Compares the linear model length change prediction
    delta_L = tau * beta * delta_b
against the full flowline model, and shows supporting diagnostics.
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

    print("Creating linear model comparison plot...")
    ds = xr.open_dataset(combined_results_path)
    delx = float(ds.attrs["delx"])

    ordered_profiles = ["hourglass", "oval", "neutral"]
    colors = {"hourglass": "purple", "oval": "teal", "neutral": "green"}

    spinup_dir = output_dir / "spinup_profiles"
    spinup_ela = {}
    spinup_ds = {}
    for profile_type in ordered_profiles:
        sp_path = spinup_dir / f"spinup_spinup_{profile_type}.nc"
        if sp_path.exists():
            sp = xr.open_dataset(sp_path)
            spinup_ds[profile_type] = sp
            if "ela" in sp:
                spinup_ela[profile_type] = float(sp["ela"].isel(time=-1))

    # --- Compute linear model quantities per profile ---
    tau = {}
    beta_abl = {}
    beta_required = {}
    delta_b = {}
    delta_L_linear = {}
    delta_L_real = {}

    for profile_type in ordered_profiles:
        if profile_type not in spinup_ds or profile_type not in ds.coords["run_id"].values:
            continue

        sp = spinup_ds[profile_type]

        h_sp = sp["h"].isel(time=-1)
        b_sp = sp["b_profile"].isel(time=-1)
        edge_idx_sp = int(sp["edge_idx"].isel(time=-1))

        ice_mask_sp = h_sp > 0
        abl_mask_sp = b_sp < 0
        combined_sp = ice_mask_sp & abl_mask_sp

        b_term_sp = float(b_sp.isel(x=edge_idx_sp))
        h_abl_mean_sp = float(h_sp.where(combined_sp).mean())

        # tau = -H / b_terminus  (positive timescale, yr)
        tau[profile_type] = -h_abl_mean_sp / b_term_sp

        h = ds["h"].sel(run_id=profile_type)
        w_ice = ds["w"].sel(run_id=profile_type)
        b = ds["b_profile"].sel(run_id=profile_type)
        edge = ds["edge"].sel(run_id=profile_type)
        total_mb = ds["total_mass_balance"].sel(run_id=profile_type)
        area = ds["area"].sel(run_id=profile_type)

        ice_mask = h.isel(time=0) > 0
        total_area_t0 = float(w_ice.where(ice_mask).sum() * delx)
        w_mean = float(w_ice.where(ice_mask).mean())
        h_mean = float(h.isel(time=0).where(ice_mask).mean())
        beta_abl[profile_type] = total_area_t0 / (w_mean * h_mean)

        delta_b[profile_type] = float((total_mb / area).isel(time=0))

        delta_L_linear[profile_type] = tau[profile_type] * beta_abl[profile_type] * delta_b[profile_type] / 1000  # m -> km
        delta_L_real[profile_type] = (float(edge.isel(time=-1)) - float(edge.isel(time=0))) / 1000
        # beta that would make linear model match the full model exactly
        beta_required[profile_type] = delta_L_real[profile_type] * 1000 / (tau[profile_type] * delta_b[profile_type])

    subplot_mosaic = [
        ["beta_ablation",  "beta_ablation",  "beta_global",      "beta_global",      "beta_length",      "beta_length",      "f_eq",          "f_eq"         ],
        ["thickness_final","thickness_final", "terminus_mb",      "terminus_mb",      "flux_pct_change",  "flux_pct_change",  "plan_hourglass","plan_hourglass"],
        ["scatter_dL",     "scatter_dL",      "scatter_tau",      "scatter_tau",      "plan_oval",        "plan_oval",        "plan_neutral",  "plan_neutral" ],
        ["length_response","length_response",  "total_flux",       "total_flux",       "flux_profiles",    "flux_profiles",    ".",              "."              ],
    ]

    fig = plt.figure(figsize=(32, 16), layout="constrained")
    axes = fig.subplot_mosaic(subplot_mosaic)
    fig.suptitle("Width Shape Variations: Linear Model Comparison", fontsize=16)

    # --- Beta plots ---
    if "h" in ds.data_vars and "w" in ds.data_vars and "b_profile" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            h = ds["h"].sel(run_id=profile_type)
            w_ice = ds["w"].sel(run_id=profile_type)
            b = ds["b_profile"].sel(run_id=profile_type)

            ice_mask = h > 0
            abl_mask = b < 0
            combined_mask = abl_mask & ice_mask

            total_area = w_ice.where(ice_mask, 0).sum(dim="x") * delx

            w_abl = w_ice.where(combined_mask).mean(dim="x")
            h_abl = h.where(combined_mask).mean(dim="x")
            beta_ablation_ts = total_area / (w_abl * h_abl)

            w_mean = w_ice.where(ice_mask).mean(dim="x")
            h_mean = h.where(ice_mask).mean(dim="x")
            beta_global_ts = total_area / (w_mean * h_mean)

            length = ds["edge"].sel(run_id=profile_type)

            label = profile_type.replace("_", " ").title()
            color = colors[profile_type]
            t = h.coords["time"]
            axes["beta_ablation"].plot(t, beta_ablation_ts, color=color, linewidth=2, label=label)
            axes["beta_global"].plot(t, beta_global_ts, color=color, linewidth=2, label=label)
            axes["beta_length"].plot(t, length / h_abl, color=color, linewidth=2, label=label)

        for ax_key, title, ylabel in [
            ("beta_ablation", "Beta: area / (w_abl * h_abl)",   "Beta (ablation zone mean)"),
            ("beta_global",   "Beta: area / (w_mean * h_mean)", "Beta (whole-glacier mean)"),
            ("beta_length",   "Beta: length / h_below_ela",     "Beta (length / mean h below ELA)"),
        ]:
            axes[ax_key].set_title(title)
            axes[ax_key].set_xlabel("Time (years)")
            axes[ax_key].set_ylabel(ylabel)
            axes[ax_key].legend()
            axes[ax_key].grid(True, alpha=0.3)

    # --- Fractional equilibration ---
    if "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            volume = (ds["h"].sel(run_id=profile_type) * ds["w"].sel(run_id=profile_type) * delx).sum(dim="x")
            V_0 = float(volume.isel(time=0))
            V_end = float(volume.isel(time=-1))
            f_eq = (volume - V_0) / (V_end - V_0)
            axes["f_eq"].plot(
                f_eq.coords["time"], f_eq,
                color=colors[profile_type], linewidth=2,
                label=profile_type.replace("_", " ").title(),
            )

        axes["f_eq"].axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        axes["f_eq"].axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        axes["f_eq"].set_xlabel("Time (years)")
        axes["f_eq"].set_ylabel("f_eq = (V - V₀) / (V_eq - V₀)")
        axes["f_eq"].set_title("Fractional Equilibration")
        axes["f_eq"].legend()
        axes["f_eq"].grid(True, alpha=0.3)

    # --- Thickness profiles ---
    if "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
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

    # --- Terminus mass balance ---
    if "b_profile" in ds.data_vars and "edge_idx" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
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

    # --- Fractional equilibration of flux/volume ratio ---
    if "F" in ds.data_vars and "h" in ds.data_vars and "w" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
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
        axes["flux_pct_change"].set_ylabel("f_flux")
        axes["flux_pct_change"].set_title("Fractional Equilibration of Flux/Volume Ratio")
        axes["flux_pct_change"].legend()
        axes["flux_pct_change"].grid(True, alpha=0.3)

    # --- Plan views ---
    plan_axes_map = {"hourglass": "plan_hourglass", "oval": "plan_oval", "neutral": "plan_neutral"}
    if "h" in ds.data_vars:
        for profile_type in ordered_profiles:
            ax = axes[plan_axes_map[profile_type]]
            color = colors[profile_type]
            if profile_type not in ds.coords["run_id"].values:
                continue
            h_data = ds["h"].sel(run_id=profile_type)
            x_vals = h_data.coords["x"].values
            w_geom = ds["w_geom_resampled"].sel(run_id=profile_type).values

            for time_idx, linestyle, time_label in [(0, "--", "Initial"), (-1, "-", "Final")]:
                h_slice = h_data.isel(time=time_idx).values
                half_w = np.where(h_slice > 0, w_geom / 2, 0)
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
            ax.set_title(f"Plan View: {profile_type.replace('_', ' ').title()}")
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)

    # --- Scatterplot: linear model vs real length change ---
    profiles_with_data = [p for p in ordered_profiles if p in delta_L_linear and p in delta_L_real]
    if profiles_with_data:
        ax_sc = axes["scatter_dL"]
        x_vals_lin = [delta_L_linear[p] for p in profiles_with_data]
        y_vals_real = [delta_L_real[p] for p in profiles_with_data]

        for p in profiles_with_data:
            ax_sc.scatter(
                delta_L_linear[p], delta_L_real[p],
                color=colors[p], s=120, zorder=5,
                label=f"{p.replace('_', ' ').title()} (β={beta_abl[p]:.1f})",
            )
            if p in beta_required:
                ax_sc.scatter(
                    delta_L_real[p], delta_L_real[p],
                    color=colors[p], s=120, zorder=5, marker="*",
                    label=f"{p.replace('_', ' ').title()} β_req={beta_required[p]:.1f}",
                )

        all_vals = x_vals_lin + y_vals_real
        v_neg = min(all_vals) * 1.1
        v_pos = max(all_vals) * 0.9
        ax_sc.plot([v_neg, v_pos], [v_neg, v_pos], "k--", linewidth=1.5, alpha=0.6, label="1:1")

        ax_sc.set_xlim(v_pos, v_neg)
        ax_sc.set_ylim(v_pos, v_neg)
        ax_sc.set_xlabel("Linear model delta_L (km)\ntau * beta * delta_b")
        ax_sc.set_ylabel("Full model delta_L (km)")
        ax_sc.set_title("Linear vs Full Model Length Change\n(circles=predicted beta, stars=required beta)")
        ax_sc.legend(fontsize=7)
        ax_sc.grid(True, alpha=0.3)

        for p in profiles_with_data:
            ax_sc.annotate(
                f"  {p.replace('_', ' ').title()}",
                (delta_L_linear[p], delta_L_real[p]),
                fontsize=8,
            )

    # --- Scatterplot: tau values ---
    if tau:
        ax_tau = axes["scatter_tau"]
        profile_labels = [p.replace("_", " ").title() for p in ordered_profiles if p in tau]
        x_pos = np.arange(len(profile_labels))

        for i, p in enumerate([p for p in ordered_profiles if p in tau]):
            ax_tau.scatter(i, tau[p], color=colors[p], s=120, zorder=5, label=p.replace("_", " ").title())

        ax_tau.set_xticks(x_pos)
        ax_tau.set_xticklabels(profile_labels, rotation=30, ha="right")
        ax_tau.set_ylabel("tau (yr)\n-H / b_terminus")
        ax_tau.set_title("Response Time Tau by Profile")
        ax_tau.legend()
        ax_tau.grid(True, alpha=0.3)

    # --- Length change over time ---
    if "edge" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            length_km = ds["edge"].sel(run_id=profile_type) / 1000
            axes["length_response"].plot(
                length_km.coords["time"], length_km,
                color=colors[profile_type], linewidth=2,
                label=profile_type.replace("_", " ").title(),
            )

        axes["length_response"].set_xlabel("Time (years)")
        axes["length_response"].set_ylabel("Length (km)")
        axes["length_response"].set_title("Length Response to +1.5°C Warming")
        axes["length_response"].legend()
        axes["length_response"].grid(True, alpha=0.3)

    # --- Total ice flux over time ---
    if "F" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            total_flux = ds["F"].sel(run_id=profile_type).sum(dim="x")
            axes["total_flux"].plot(
                total_flux.coords["time"], total_flux,
                color=colors[profile_type], linewidth=2,
                label=profile_type.replace("_", " ").title(),
            )

        axes["total_flux"].set_xlabel("Time (years)")
        axes["total_flux"].set_ylabel("Total ice flux (m³/yr)")
        axes["total_flux"].set_title("Total Ice Flux Over Time")
        axes["total_flux"].legend()
        axes["total_flux"].grid(True, alpha=0.3)

    # --- Initial and final flux profiles ---
    if "F" in ds.data_vars:
        for profile_type in ordered_profiles:
            if profile_type not in ds.coords["run_id"].values:
                continue
            F_data = ds["F"].sel(run_id=profile_type)
            x_km = F_data.coords["x"] / 1000
            label = profile_type.replace("_", " ").title()
            axes["flux_profiles"].plot(x_km, F_data.isel(time=0), color=colors[profile_type], linewidth=2, linestyle="--")
            axes["flux_profiles"].plot(x_km, F_data.isel(time=-1), color=colors[profile_type], linewidth=2, label=label)

        axes["flux_profiles"].set_xlabel("Distance from head (km)")
        axes["flux_profiles"].set_ylabel("Ice flux (m³/yr)")
        axes["flux_profiles"].set_title("Ice Flux Profile (solid=final, dashed=initial)")
        axes["flux_profiles"].legend()
        axes["flux_profiles"].grid(True, alpha=0.3)

    plot_path = output_dir / "width_shape_linear_model.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Linear model comparison plot saved to: {plot_path}")

    print("\nLinear model summary:")
    for p in profiles_with_data:
        print(f"  {p}:")
        print(f"    tau        = {tau[p]:.1f} yr")
        print(f"    beta       = {beta_abl[p]:.2f}")
        print(f"    delta_b    = {delta_b[p]:.4f} m/yr")
        print(f"    delta_L (linear) = {delta_L_linear[p]:.2f} km")
        print(f"    delta_L (full)   = {delta_L_real[p]:.2f} km")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "output")
    plot(output_dir)
