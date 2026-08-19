#!/usr/bin/env python3
"""
Plots for the width_shape_grid_search example.

Produces:
  1. scatter_T0_vs_length.png  — steady-state length vs temperature for each shape
  2. branch_{shape}.png        — per-shape diagnostics: length fractional equilibration,
                                  length change, flux, volume, and linear model comparison
"""

import csv
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit


SHAPES = ["hourglass", "oval", "neutral"]
SHAPE_COLORS = {"hourglass": "purple", "oval": "orange", "neutral": "green"}
DT_VALUES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def make_run_id(shape: str, dT: float) -> str:
    sign = "p" if dT >= 0 else "m"
    return f"{shape}_dT_{sign}{abs(dT):.2f}"


def dT_colormap():
    n = len(DT_VALUES)
    base_colors = plt.get_cmap("RdBu_r")(np.linspace(0, 1, n))
    cmap = mcolors.ListedColormap(base_colors)
    bounds = np.array(DT_VALUES + [DT_VALUES[-1] + (DT_VALUES[-1] - DT_VALUES[-2])])
    bounds -= (bounds[1] - bounds[0]) / 2
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _get_steady_state_length(ds, run_id):
    """Return final edge length (m) for a run."""
    return float(ds["edge"].sel(run_id=run_id).isel(time=-1))


def _load_spinup(spinup_dir, shape):
    # Spinup files are named spinup_spinup_{run_id}.nc (double prefix: one from
    # FlowlineSpinup.generate_profile prepending "spinup_", one from entrypoints).
    # The run_id used is the first dict key for each unique spinup object, which
    # is {shape}_dT_m1.00 since DT_VALUES starts at -1.0.
    candidates = sorted(spinup_dir.glob(f"spinup_spinup_{shape}_dT_*.nc"))
    if not candidates:
        return None
    return xr.open_dataset(candidates[0])


def _beta_abl(h_t, w_ice, b_t, delx):
    """Beta using only the ablation zone (ice & b<0). Returns float or nan."""
    ice_mask = h_t > 0
    combined = ice_mask & (b_t < 0)
    total_area = float(w_ice.where(ice_mask, 0).sum() * delx)
    w_abl = float(w_ice.where(combined).mean())
    h_abl = float(h_t.where(combined).mean())
    if w_abl > 0 and h_abl > 0:
        return total_area / (w_abl * h_abl)
    return np.nan


def _beta_whole(h_t, w_ice, delx):
    """Beta using the whole glacier (no ablation mask). Returns float or nan."""
    ice_mask = h_t > 0
    total_area = float(w_ice.where(ice_mask, 0).sum() * delx)
    w_ice_mean = float(w_ice.where(ice_mask).mean())
    h_ice_mean = float(h_t.where(ice_mask).mean())
    if w_ice_mean > 0 and h_ice_mean > 0:
        return total_area / (w_ice_mean * h_ice_mean)
    return np.nan


def compute_linear_model_quantities(ds, spinup_ds, shape, run_id, delx):
    """
    Compute tau, betas, delta_b, and predicted/actual delta_L for one run.

    Quantities that describe the pre-perturbation glacier state (tau, delta_b, betas
    at t=0) are drawn from the spinup final state, not the experiment's t=0, because
    b_profile changes in the first timestep of the experiment run.

    Returns dict with keys:
      tau, tau_mean_h, beta_init_abl, beta_final_abl, beta_init_whole,
      delta_b, delta_L_linear, delta_L_real, beta_required
    """
    sp = spinup_ds
    h_sp = sp["h"].isel(time=-1)
    b_sp = sp["b_profile"].isel(time=-1)
    w_sp = sp["w"].isel(time=-1) if "time" in sp["w"].dims else sp["w"]
    edge_idx_sp = int(sp["edge_idx"].isel(time=-1))
    area_sp = float(sp["area"].isel(time=-1))
    total_mb_sp = float(sp["total_mass_balance"].isel(time=-1))

    ice_mask_sp = h_sp > 0
    combined_sp = ice_mask_sp & (b_sp < 0)

    b_term_sp = float(b_sp.isel(x=edge_idx_sp))
    h_abl_mean_sp = float(h_sp.where(combined_sp).mean())
    h_mean_sp = float(h_sp.where(ice_mask_sp).mean())
    h_max_sp = float(h_sp.where(ice_mask_sp).max())

    # tau from ablation-zone mean thickness: tau = -H_abl / b_terminus
    tau = -h_abl_mean_sp / b_term_sp

    # tau from whole-glacier mean thickness: tau = -H_mean / b_terminus
    tau_mean_h = -h_mean_sp / b_term_sp

    # tau from max thickness: tau = -H_max / b_terminus
    tau_max_h = -h_max_sp / b_term_sp

    h = ds["h"].sel(run_id=run_id)
    w_ice = ds["w"].sel(run_id=run_id)
    b = ds["b_profile"].sel(run_id=run_id)
    edge = ds["edge"].sel(run_id=run_id)
    total_mb = ds["total_mass_balance"].sel(run_id=run_id)
    area = ds["area"].sel(run_id=run_id)

    # Use spinup final state for h, w, and ablation mask (b_profile shifts at experiment t=0)
    beta_init_abl = _beta_abl(h_sp, w_sp, b_sp, delx)
    beta_final_abl = _beta_abl(h.isel(time=-1), w_ice, b.isel(time=-1), delx)
    beta_init_whole = _beta_whole(h_sp, w_ice, delx)

    # delta_b: perturbation to specific mass balance, taken from experiment t=0
    # (this is the forcing change itself, not the pre-perturbation state)
    delta_b = float((total_mb / area).isel(time=0))

    delta_L_linear = tau * beta_init_abl * delta_b / 1000  # m -> km
    delta_L_real = (float(edge.isel(time=-1)) - float(edge.isel(time=0))) / 1000

    beta_required = delta_L_real * 1000 / (tau * delta_b) if abs(tau * delta_b) > 1e-10 else np.nan

    return {
        "tau": tau,
        "tau_mean_h": tau_mean_h,
        "tau_max_h": tau_max_h,
        "beta": beta_init_abl,        # kept for backwards compat with scatter plot
        "beta_init_abl": beta_init_abl,
        "beta_final_abl": beta_final_abl,
        "beta_init_whole": beta_init_whole,
        "delta_b": delta_b,
        "delta_L_linear": delta_L_linear,
        "delta_L_real": delta_L_real,
        "beta_required": beta_required,
    }


def compute_weighted_beta(ds, run_id, delx, whole_glacier=False):
    """
    Compute a weighted-mean beta where each timestep's beta is weighted by
    the incremental length change to the next timestep.

    If whole_glacier=False (default): total area / (mean w_abl * mean h_abl)
    If whole_glacier=True: total area / (mean w_ice * mean h_ice)
    Returns weighted_beta (float).
    """
    h = ds["h"].sel(run_id=run_id)
    w_ice = ds["w"].sel(run_id=run_id)
    b = ds["b_profile"].sel(run_id=run_id)
    edge = ds["edge"].sel(run_id=run_id)

    n_times = len(h.coords["time"])
    betas = []
    weights = []

    for ti in range(n_times - 1):
        ice_mask = h.isel(time=ti) > 0
        total_area = float(w_ice.where(ice_mask, 0).sum() * delx)

        if whole_glacier:
            w_mean = float(w_ice.where(ice_mask).mean())
            h_mean = float(h.where(ice_mask).isel(time=ti).mean())
            beta_t = total_area / (w_mean * h_mean) if (w_mean > 0 and h_mean > 0) else np.nan
        else:
            abl_mask = b.isel(time=ti) < 0
            combined = ice_mask & abl_mask
            w_abl = float(w_ice.where(combined).mean())
            h_abl = float(h.where(combined).isel(time=ti).mean())
            beta_t = total_area / (w_abl * h_abl) if (w_abl > 0 and h_abl > 0) else np.nan

        dL = abs(float(edge.isel(time=ti + 1)) - float(edge.isel(time=ti)))
        betas.append(beta_t)
        weights.append(dL)

    betas = np.array(betas)
    weights = np.array(weights)
    valid = np.isfinite(betas) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(betas[valid], weights=weights[valid]))


def compute_weighted_tau(ds, run_id):
    """
    Compute a time-mean tau = -H_abl / b_terminus, averaged over all timesteps.
    """
    h = ds["h"].sel(run_id=run_id)
    b = ds["b_profile"].sel(run_id=run_id)
    edge_idx = ds["edge_idx"].sel(run_id=run_id)

    n_times = len(h.coords["time"])
    taus = []

    for ti in range(n_times):
        ice_mask = h.isel(time=ti) > 0
        abl_mask = b.isel(time=ti) < 0
        combined = ice_mask & abl_mask

        h_abl = float(h.where(combined).isel(time=ti).mean())
        b_term = float(b.isel(time=ti, x=int(edge_idx.isel(time=ti))))
        tau_t = -h_abl / b_term if (np.isfinite(h_abl) and np.isfinite(b_term) and abs(b_term) > 1e-10) else np.nan
        taus.append(tau_t)

    taus = np.array(taus)
    valid = np.isfinite(taus)
    if not valid.any():
        return np.nan
    return float(np.mean(taus[valid]))


def fit_exponential(t, y):
    """Fit y = 1 - exp(-t/tau) to fractional equilibration. Returns tau or nan."""
    def model(t, tau):
        return 1 - np.exp(-t / tau)
    try:
        popt, _ = curve_fit(model, t, y, p0=[50.0], bounds=(1e-3, 1e6))
        return float(popt[0])
    except Exception:
        return np.nan


def fit_three_stage_length(t, y):
    """
    Fit fractional length equilibration to:
      f_eq(t) = 1 - exp(-t*sqrt(3)/tau) * (1 + t*sqrt(3)/tau + (1/2)*(t*sqrt(3)/tau)^2)
    Returns tau or nan.
    """
    def model(t, tau):
        x = t * np.sqrt(3) / tau
        return 1 - np.exp(-x) * (1 + x + 0.5 * x**2)
    # skip the initial flat/quantized region before the glacier makes its first discrete grid step
    t_fit = t[1:]
    y_fit = y[1:]
    first_nonzero = np.argmax(y_fit > 0)
    t_fit = t_fit[first_nonzero:]
    y_fit = y_fit[first_nonzero:]
    if len(t_fit) < 4:
        return np.nan
    try:
        popt, _ = curve_fit(model, t_fit, y_fit, p0=[50.0], bounds=(1e-3, 1e6))
        return float(popt[0])
    except Exception:
        return np.nan


def _read_spinup_T0(spinup_dir, shape):
    """Extract optimized T0 from spinup profile attributes (stored as forcing_T0)."""
    sp = _load_spinup(spinup_dir, shape)
    if sp is None:
        return None
    if "forcing_T0" in sp.attrs:
        return float(sp.attrs["forcing_T0"])
    return None


def plot_scatter_T0_vs_length(ds, spinup_dir, output_dir):
    """Scatter: equilibrium T0 (spinup T0 + dT) vs steady-state length, one line per shape.
    Second subplot shifts each line so the spinup T0 maps to 0 and the spinup length to 8 km,
    putting all shapes on a common reference frame."""
    fig, axes = plt.subplot_mosaic([["abs", "rel"]], figsize=(16, 6))

    have_absolute_T0 = False
    shape_data = {}
    for shape in SHAPES:
        T0_spinup = _read_spinup_T0(spinup_dir, shape)
        if T0_spinup is not None:
            have_absolute_T0 = True

        T0_vals = []
        length_vals = []
        for dT in DT_VALUES:
            run_id = make_run_id(shape, dT)
            if run_id not in ds.coords["run_id"].values:
                continue
            x_val = dT if T0_spinup is None else T0_spinup + dT
            T0_vals.append(x_val)
            length_vals.append(_get_steady_state_length(ds, run_id) / 1000)

        shape_data[shape] = {"T0_vals": T0_vals, "length_vals": length_vals, "T0_spinup": T0_spinup}

    TARGET_LENGTH_KM = 8.0

    for shape, data in shape_data.items():
        label = shape.replace("_", " ").title()
        color = SHAPE_COLORS[shape]
        T0_vals = data["T0_vals"]
        length_vals = data["length_vals"]
        T0_spinup = data["T0_spinup"]

        axes["abs"].plot(T0_vals, length_vals, "-o", color=color, linewidth=2, markersize=6, label=label)

        if len(T0_vals) >= 2:
            m_abs, b_abs = np.polyfit(T0_vals, length_vals, 1)
            x_fit = np.linspace(min(T0_vals), max(T0_vals), 100)
            axes["abs"].plot(x_fit, np.polyval([m_abs, b_abs], x_fit), "--", color=color, linewidth=1, alpha=0.6)
            sign = "+" if b_abs >= 0 else "-"
            axes["abs"].annotate(
                f"L = {m_abs:.2f}T {sign} {abs(b_abs):.2f}",
                xy=(x_fit[-1], np.polyval([m_abs, b_abs], x_fit[-1])),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=color, va="center",
            )

        # Shift x so spinup T0 -> 0, y so spinup length -> TARGET_LENGTH_KM
        T0_ref = T0_spinup if T0_spinup is not None else 0.0
        length_ref = np.interp(T0_ref, T0_vals, length_vals) if T0_vals else TARGET_LENGTH_KM
        rel_T0 = [v - T0_ref for v in T0_vals]
        rel_length = [v - length_ref + TARGET_LENGTH_KM for v in length_vals]
        axes["rel"].plot(rel_T0, rel_length, "-o", color=color, linewidth=2, markersize=6, label=label)

        if len(rel_T0) >= 2:
            m_rel, b_rel = np.polyfit(rel_T0, rel_length, 1)
            x_fit_rel = np.linspace(min(rel_T0), max(rel_T0), 100)
            axes["rel"].plot(x_fit_rel, np.polyval([m_rel, b_rel], x_fit_rel), "--", color=color, linewidth=1, alpha=0.6)
            sign = "+" if b_rel >= 0 else "-"
            axes["rel"].annotate(
                f"L = {m_rel:.2f}ΔT {sign} {abs(b_rel):.2f}",
                xy=(x_fit_rel[-1], np.polyval([m_rel, b_rel], x_fit_rel[-1])),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=color, va="center",
            )

    xlabel_abs = "Temperature T0 (°C)" if have_absolute_T0 else "dT relative to spinup T0 (°C)"
    axes["abs"].set_xlabel(xlabel_abs)
    axes["abs"].set_ylabel("Steady-state length (km)")
    axes["abs"].set_title("Steady-State Length vs Temperature")
    axes["abs"].legend()
    axes["abs"].grid(True, alpha=0.3)

    axes["rel"].axvline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    axes["rel"].axhline(TARGET_LENGTH_KM, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    axes["rel"].set_xlabel("Relative temperature change (°C)")
    axes["rel"].set_ylabel("Steady-state length (km)")
    axes["rel"].set_title("Steady-State Length vs Relative Temperature\n(aligned to spinup T0 and 8 km)")
    axes["rel"].legend()
    axes["rel"].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "scatter_T0_vs_length.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_branch(ds, spinup_dir, shape, delx, output_dir):
    """Per-shape figure: length fractional equilibration, length change, total flux, volume, and linear model scatter."""
    cmap, norm = dT_colormap()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    mosaic = [
        ["length_frac", "length", "flux", "volume"],
        ["lm_scatter", "lm_scatter", "lm_scatter", "lm_resid"],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 10))
    fig.suptitle(f"Width Shape: {shape.replace('_', ' ').title()}", fontsize=14)

    sp = _load_spinup(spinup_dir, shape)

    lm_actual = []
    lm_predicted = []
    lm_dT_colors = []

    # Collections for diagnostics printing and curve fitting
    lm_results = {}        # dT -> lm dict from compute_linear_model_quantities
    weighted_betas = {}    # dT -> weighted_beta
    weighted_taus = {}     # dT -> tau_wtd
    frac_L_series = {}             # dT -> (t, frac_L array)
    frac_flux_series = {}          # dT -> (t, frac_flux array)
    frac_vol_series = {}           # dT -> (t, frac_vol array)
    frac_flux_per_vol_series = {}  # dT -> (t, frac of total_flux/volume)

    for dT in DT_VALUES:
        run_id = make_run_id(shape, dT)
        if run_id not in ds.coords["run_id"].values:
            continue

        color = cmap(norm(dT))

        # --- Length fractional equilibration ---
        edge = ds["edge"].sel(run_id=run_id)
        t = edge.coords["time"].values
        if dT != 0.0:
            delta_L = edge.values - edge.values[0]
            delta_L_eq = edge.values[-1] - edge.values[0]
            if abs(delta_L_eq) > 1e-10:
                frac_L = delta_L / delta_L_eq
                axes["length_frac"].plot(t, frac_L, color=color, linewidth=1.5)
                frac_L_series[dT] = (t, frac_L)

        # --- Length change magnitude over time ---
        delta_L_km = np.abs((edge.values - edge.values[0]) / 1000)
        axes["length"].plot(t, delta_L_km, color=color, linewidth=1.5, label=f"{dT:+.2f}")

        # --- Total flux fractional equilibration ---
        if "F" in ds.data_vars and dT != 0.0:
            total_flux = ds["F"].sel(run_id=run_id).sum(dim="x").values
            flux_range = total_flux[-1] - total_flux[0]
            if abs(flux_range) > 1e-10:
                frac_flux = (total_flux - total_flux[0]) / flux_range
                axes["flux"].plot(t, frac_flux, color=color, linewidth=1.5)
                frac_flux_series[dT] = (t, frac_flux)

        # --- Volume fractional equilibration ---
        if "h" in ds.data_vars and "w" in ds.data_vars and dT != 0.0:
            volume = (ds["h"].sel(run_id=run_id) * ds["w"].sel(run_id=run_id) * delx).sum(dim="x").values
            vol_range = volume[-1] - volume[0]
            if abs(vol_range) > 1e-10:
                frac_vol = (volume - volume[0]) / vol_range
                axes["volume"].plot(t, frac_vol, color=color, linewidth=1.5)
                frac_vol_series[dT] = (t, frac_vol)

            # --- Flux-per-volume fractional equilibration ---
            if "F" in ds.data_vars and abs(vol_range) > 1e-10:
                total_flux = ds["F"].sel(run_id=run_id).sum(dim="x").values
                with np.errstate(divide="ignore", invalid="ignore"):
                    flux_per_vol = np.where(volume > 0, total_flux / volume, np.nan)
                fpv_range = flux_per_vol[-1] - flux_per_vol[0]
                if np.isfinite(fpv_range) and abs(fpv_range) > 1e-10:
                    frac_fpv = (flux_per_vol - flux_per_vol[0]) / fpv_range
                    frac_flux_per_vol_series[dT] = (t, frac_fpv)

        # --- Linear model quantities ---
        if sp is not None and "h" in ds.data_vars:
            lm = compute_linear_model_quantities(ds, sp, shape, run_id, delx)
            lm_results[dT] = lm
            lm_actual.append(lm["delta_L_real"])
            lm_predicted.append(lm["delta_L_linear"])
            lm_dT_colors.append(dT)

            if dT != 0.0:
                wb = compute_weighted_beta(ds, run_id, delx)
                wb_whole = compute_weighted_beta(ds, run_id, delx, whole_glacier=True)
                weighted_betas[dT] = (wb, wb_whole)
                weighted_taus[dT] = compute_weighted_tau(ds, run_id)

    # Finalize time-series axes
    axes["length_frac"].axhline(1, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    axes["length_frac"].set_xlabel("Time (years)")
    axes["length_frac"].set_ylabel("Fractional equilibration")
    axes["length_frac"].set_title("Length Fractional Equilibration")
    axes["length_frac"].set_ylim(0, 1)
    axes["length_frac"].grid(True, alpha=0.3)

    axes["length"].set_xlabel("Time (years)")
    axes["length"].set_ylabel("Length change magnitude (km)")
    axes["length"].set_title("Length Change Magnitude Over Time")
    axes["length"].grid(True, alpha=0.3)

    axes["flux"].axhline(1, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    axes["flux"].set_xlabel("Time (years)")
    axes["flux"].set_ylabel("Fractional equilibration")
    axes["flux"].set_title("Total Ice Flux Equilibration")
    axes["flux"].set_ylim(0, 1)
    axes["flux"].grid(True, alpha=0.3)

    axes["volume"].axhline(1, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    axes["volume"].set_xlabel("Time (years)")
    axes["volume"].set_ylabel("Fractional equilibration")
    axes["volume"].set_title("Ice Volume Equilibration")
    axes["volume"].set_ylim(0, 1)
    axes["volume"].grid(True, alpha=0.3)

    # Add discrete colorbar for dT to length panel
    fig.colorbar(sm, ax=axes["length"], label="dT (°C)", ticks=DT_VALUES)

    # --- Linear model scatter ---
    ax_lm = axes["lm_scatter"]
    if lm_actual and lm_predicted:
        sc_colors = [cmap(norm(dT)) for dT in lm_dT_colors]
        for x, y, c, dT in zip(lm_actual, lm_predicted, sc_colors, lm_dT_colors):
            ax_lm.scatter(x, y, color=c, s=80, zorder=5)
            ax_lm.annotate(f" {dT:+.2f}", (x, y), fontsize=7, va="center")

        all_vals = lm_actual + lm_predicted
        vmin, vmax = min(all_vals), max(all_vals)
        pad = (vmax - vmin) * 0.1 if vmax != vmin else 0.5
        lims = (vmin - pad, vmax + pad)
        ax_lm.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6, label="1:1")
        ax_lm.set_xlim(lims)
        ax_lm.set_ylim(lims)

    ax_lm.set_xlabel("Full model delta_L (km)")
    ax_lm.set_ylabel("Linear model delta_L (km)\ntau * beta * delta_b")
    ax_lm.set_title("Linear Model vs Full Model: Steady-State Length Change")
    ax_lm.legend(fontsize=8)
    ax_lm.grid(True, alpha=0.3)

    # --- Residuals to 1:1 line ---
    ax_res = axes["lm_resid"]
    if lm_actual and lm_predicted:
        sc_colors = [cmap(norm(dT)) for dT in lm_dT_colors]
        residuals = [y - x for x, y in zip(lm_actual, lm_predicted)]
        for x, r, c, dT in zip(lm_actual, residuals, sc_colors, lm_dT_colors):
            ax_res.scatter(x, r, color=c, s=80, zorder=5)
            ax_res.annotate(f" {dT:+.2f}", (x, r), fontsize=7, va="center")
        ax_res.axhline(0, color="k", linewidth=1.2, linestyle="--", alpha=0.6)

    ax_res.set_xlabel("Full model delta_L (km)")
    ax_res.set_ylabel("Residual (linear - full) (km)")
    ax_res.set_title("Residuals to 1:1 Line")
    ax_res.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"branch_{shape}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- Print diagnostics and return rows for CSV ---
    return _print_diagnostics(shape, lm_results, weighted_betas, weighted_taus, frac_L_series, frac_flux_series, frac_vol_series, frac_flux_per_vol_series)


def _print_diagnostics(shape, lm_results, weighted_betas, weighted_taus, frac_L_series, frac_flux_series, frac_vol_series, frac_flux_per_vol_series):
    """Print a table of linear model and curve-fit diagnostics for one shape.
    Returns a list of dicts (one per dT row) for CSV export."""
    print(f"\n{'='*130}")
    print(f"Diagnostics: {shape.replace('_', ' ').title()}")
    print(f"{'='*130}")

    header = (
        f"{'dT':>6}  {'dL_real':>9}  {'dL_init_abl':>12}  {'dL_final_abl':>13}  {'dL_whole':>9}  {'dL_wtd_beta':>12}"
        f"  {'tau_req_init':>13}  {'tau_req_wtd':>12}  {'tau_req_whole':>14}"
        f"  {'tau_flux_exp':>13}  {'tau_vol_exp':>12}  {'tau_len_3st':>12}  {'tau_93pct':>10}  {'tau_mean_h':>11}  {'tau_max_h':>10}"
    )
    print(header)
    print("-" * len(header))

    def _fmt(v):
        return f"{v:>9.2f}" if np.isfinite(v) else f"{'nan':>9}"

    rows = []
    for dT in sorted(lm_results.keys()):
        if dT == 0.0:
            continue
        lm = lm_results[dT]
        tau = lm["tau"]
        tau_mean_h = lm["tau_mean_h"]
        beta_init_abl = lm["beta_init_abl"]
        beta_final_abl = lm["beta_final_abl"]
        beta_init_whole = lm["beta_init_whole"]
        delta_b = lm["delta_b"]
        delta_L_real = lm["delta_L_real"]

        delta_L_init_abl = tau * beta_init_abl * delta_b / 1000 if np.isfinite(beta_init_abl) else np.nan
        delta_L_final_abl = tau * beta_final_abl * delta_b / 1000 if np.isfinite(beta_final_abl) else np.nan
        delta_L_whole = tau * beta_init_whole * delta_b / 1000 if np.isfinite(beta_init_whole) else np.nan

        wb, wb_whole = weighted_betas.get(dT, (np.nan, np.nan))
        delta_L_wtd = tau * wb * delta_b / 1000 if np.isfinite(wb) else np.nan
        delta_L_wtd_whole = tau * wb_whole * delta_b / 1000 if np.isfinite(wb_whole) else np.nan

        denom_init = beta_init_abl * delta_b
        tau_req_init = delta_L_real * 1000 / denom_init if (np.isfinite(beta_init_abl) and abs(denom_init) > 1e-10) else np.nan

        denom_wtd = wb * delta_b
        tau_req_wtd = delta_L_real * 1000 / denom_wtd if (np.isfinite(wb) and abs(denom_wtd) > 1e-10) else np.nan

        denom_whole = beta_init_whole * delta_b
        tau_req_whole = delta_L_real * 1000 / denom_whole if (np.isfinite(beta_init_whole) and abs(denom_whole) > 1e-10) else np.nan

        tau_flux_exp = np.nan
        if dT in frac_flux_series:
            t_f, y_f = frac_flux_series[dT]
            tau_flux_exp = fit_exponential(t_f, y_f)

        tau_vol_exp = np.nan
        if dT in frac_vol_series:
            t_v, y_v = frac_vol_series[dT]
            tau_vol_exp = fit_exponential(t_v, y_v)

        tau_flux_per_vol_exp = np.nan
        if dT in frac_flux_per_vol_series:
            t_fpv, y_fpv = frac_flux_per_vol_series[dT]
            valid = np.isfinite(y_fpv)
            if valid.sum() > 3:
                tau_flux_per_vol_exp = fit_exponential(t_fpv[valid], y_fpv[valid])

        tau_len_lm = np.nan
        if dT in frac_L_series:
            t_l, y_l = frac_L_series[dT]
            tau_len_lm = fit_three_stage_length(t_l, y_l)

        tau_93pct = np.nan
        if dT in frac_L_series:
            t_l, y_l = frac_L_series[dT]
            idx = np.searchsorted(y_l, 0.938)
            if 0 < idx < len(t_l):
                tau_93pct = t_l[idx] / 3.0

        tau_max_h = lm["tau_max_h"]
        tau_wtd = weighted_taus.get(dT, np.nan)
        print(
            f"{dT:>+6.2f}  {delta_L_real:>9.3f}  {_fmt(delta_L_init_abl)}     {_fmt(delta_L_final_abl)}     {_fmt(delta_L_whole)}  {_fmt(delta_L_wtd)}"
            f"  {_fmt(tau_req_init)}     {_fmt(tau_req_wtd)}   {_fmt(tau_req_whole)}"
            f"  {_fmt(tau_flux_exp)}     {_fmt(tau_vol_exp)}  {_fmt(tau_len_lm)}  {_fmt(tau_93pct)}  {_fmt(tau_mean_h)}  {_fmt(tau_max_h)}  {_fmt(tau_wtd)}"
        )

        rows.append({
            "shape": shape,
            "dT": dT,
            "dL_real": delta_L_real,
            "dL_init_abl": delta_L_init_abl,
            "dL_final_abl": delta_L_final_abl,
            "dL_whole": delta_L_whole,
            "dL_wtd_beta": delta_L_wtd,
            "dL_wtd_beta_whole": delta_L_wtd_whole,
            "delta_b": delta_b,
            "tau": lm["tau"],
            "tau_wtd": tau_wtd,
            "tau_req_init": tau_req_init,
            "tau_req_wtd": tau_req_wtd,
            "tau_req_whole": tau_req_whole,
            "tau_flux_exp": tau_flux_exp,
            "tau_vol_exp": tau_vol_exp,
            "tau_flux_per_vol_exp": tau_flux_per_vol_exp,
            "tau_len_3st": tau_len_lm,
            "tau_93pct": tau_93pct,
            "tau_mean_h": tau_mean_h,
            "tau_max_h": tau_max_h,
            "beta_init_abl": beta_init_abl,
            "beta_final_abl": beta_final_abl,
            "beta_init_whole": beta_init_whole,
            "beta_wtd": wb,
            "beta_wtd_whole": wb_whole,
        })

    print()
    print("Column descriptions:")
    print("  dL_real       : full model equilibrium length change (km)")
    print("  dL_init_abl   : linear model: tau * beta_init_abl * delta_b  (beta over ablation zone, t=0)")
    print("  dL_final_abl  : linear model: tau * beta_final_abl * delta_b (beta over ablation zone, t=end)")
    print("  dL_whole      : linear model: tau * beta_init_whole * delta_b (beta over whole glacier, t=0)")
    print("  dL_wtd_beta   : linear model: tau * beta_weighted * delta_b  (length-change-weighted mean abl beta)")
    print("  tau_req_init  : tau required for linear model (init abl beta) to match full model (yr)")
    print("  tau_req_wtd   : tau required for linear model (wtd abl beta) to match full model (yr)")
    print("  tau_req_whole : tau required for linear model (whole-glacier beta) to match full model (yr)")
    print("  tau_flux_exp  : e-folding tau from exponential fit to flux fractional equilibration (yr)")
    print("  tau_vol_exp   : e-folding tau from exponential fit to volume fractional equilibration (yr)")
    print("  tau_len_3st   : tau from 3-stage linear model fit to length fractional equilibration (yr), eps=1/sqrt(3)")
    print("  tau_93pct     : tau estimated as t(93.8% equilibration) / 3  (yr)")
    print("  tau_mean_h    : tau = -mean_ice_thickness / b_terminus, from spinup final state (yr)")
    print("  tau_max_h     : tau = -max_ice_thickness / b_terminus, from spinup final state (yr)")
    print("  tau_wtd       : tau = -H_abl / b_terminus, length-change-weighted mean over experiment (yr)")

    return rows


BETA_VARIANTS = [
    ("dL_init_abl",    "init abl: tau * beta(total area / w_abl * h_abl, t=0) * db",     "steelblue"),
    ("dL_final_abl",   "final abl: same as init abl but beta at t=end",                   "firebrick"),
    ("dL_whole",       "init whole: tau * beta(total area / w * h, t=0) * db",            "seagreen"),
    ("dL_wtd_beta",    "wtd abl: tau * beta(length-change-weighted abl mean) * db",       "darkorange"),
    ("dL_wtd_beta_whole", "wtd whole: tau * beta(length-change-weighted whole mean) * db","purple"),
]

TAU_VARIANTS = [
    ("tau",          r"abl $\bar{h}$: $-\bar{h}_{abl}$ / $b_{term}$ from spinup",   "steelblue"),
    ("tau_mean_h",   r"mean h: $-\bar{h}_{ice}$ / $b_{term}$ from spinup",           "firebrick"),
    ("tau_max_h",    r"max h: $-h_{max}$ / $b_{term}$ from spinup",                  "seagreen"),
    ("tau_flux_exp",         "flux exp: e-fold fit to total flux equilibration",              "darkorange"),
    ("tau_vol_exp",          "vol exp: e-fold fit to volume equilibration",                   "purple"),
    ("tau_flux_per_vol_exp", "flux/vol exp: e-fold fit to (total flux / volume) equilibration", "crimson"),
    ("tau_len_3st",  "len 3-stage: 3-stage model fit to length equilibration",        "saddlebrown"),
    ("tau_93pct",    "93.8pct: t(93.8% length equil.) / 3",                          "teal"),
    ("tau_req_init", "req init abl: tau needed for linear model to match (init abl beta)", "gold"),
    ("tau_req_wtd",  "req wtd: tau needed for linear model to match (wtd abl beta)",  "hotpink"),
    ("tau_req_whole","req whole: tau needed for linear model to match (whole beta)",   "slategray"),
]


def plot_beta_tau_table(all_rows, output_dir):
    """
    Two plots: one for beta variants, one for tau variants.
    Rows = profile shape. Colors distinguish variants within each panel.
    Tau plot has a single shared legend placed outside the subplots.
    """
    by_shape = {shape: [] for shape in SHAPES}
    for row in all_rows:
        by_shape[row["shape"]].append(row)

    # --- beta plot: rows = shape, colors = variant ---
    n = len(SHAPES)
    fig, axes = plt.subplot_mosaic([[s] for s in SHAPES], figsize=(9, 4 * n))
    fig.suptitle(r"$\beta$ variants by shape", fontsize=13)

    for shape in SHAPES:
        ax = axes[shape]
        rows_s = sorted(by_shape[shape], key=lambda r: r["dT"])
        dTs = np.array([r["dT"] for r in rows_s])
        dL_real = np.array([r.get("dL_real", np.nan) for r in rows_s], dtype=float)
        ax.plot(dTs, dL_real, "k-", linewidth=2, label="dL real (full model)")
        for key, label, color in BETA_VARIANTS:
            vals = np.array([r.get(key, np.nan) for r in rows_s], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(dTs, vals, "-o", color=color, linewidth=1.5, markersize=5, label=label)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(shape.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(r"$\Delta L$ (km)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    axes[SHAPES[-1]].set_xlabel("dT (°C)")
    fig.tight_layout()
    beta_path = output_dir / "beta_table.png"
    fig.savefig(beta_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {beta_path}")

    # --- tau plot: rows = method, colors = shape ---
    n_tau = len(TAU_VARIANTS)
    fig, axes = plt.subplot_mosaic([[key] for key, _, _ in TAU_VARIANTS], figsize=(9, 3.5 * n_tau))
    fig.suptitle(r"$\tau$ variants by method", fontsize=13)

    for key, label, _ in TAU_VARIANTS:
        ax = axes[key]
        for shape in SHAPES:
            rows_s = sorted(by_shape[shape], key=lambda r: r["dT"])
            dTs = np.array([r["dT"] for r in rows_s])
            vals = np.array([r.get(key, np.nan) for r in rows_s], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(dTs, vals, "-o", color=SHAPE_COLORS[shape], linewidth=1.5, markersize=5,
                    label=shape.replace("_", " ").title())
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel(r"$\tau$ (yr)")
        ax.grid(True, alpha=0.3)

    # single shared legend outside
    handles, labels = axes[TAU_VARIANTS[0][0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, framealpha=0.9, title="Shape", title_fontsize=9)
    axes[TAU_VARIANTS[-1][0]].set_xlabel("dT (°C)")
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    tau_path = output_dir / "tau_table.png"
    fig.savefig(tau_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tau_path}")


def plot(output_dir):
    output_dir = Path(output_dir)
    combined_path = output_dir / "combined_results.nc"

    if not combined_path.exists():
        raise FileNotFoundError(f"No combined_results.nc found in {output_dir}")

    ds = xr.open_dataset(combined_path)
    delx = float(ds.attrs["delx"])
    spinup_dir = output_dir / "spinup_profiles"

    run_ids = list(ds.coords["run_id"].values)
    print(f"Loaded {len(run_ids)} runs from combined_results.nc")

    plot_scatter_T0_vs_length(ds, spinup_dir, output_dir)

    all_rows = []
    for shape in SHAPES:
        shape_runs = [r for r in run_ids if r.startswith(shape)]
        if not shape_runs:
            print(f"No runs found for shape '{shape}', skipping.")
            continue
        print(f"Plotting branch figure for '{shape}' ({len(shape_runs)} runs)...")
        rows = plot_branch(ds, spinup_dir, shape, delx, output_dir)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        plot_beta_tau_table(all_rows, output_dir)

        csv_path = output_dir / "diagnostics.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "output")
    plot(output_dir)
