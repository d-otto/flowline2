#!/usr/bin/env python3
"""
Spinup sweep example using FlowlineSpinup objects.

Demonstrates a parameter sweep where each melt factor (mu) gets its own
FlowlineSpinup object to reach steady state before the main experimental run.
"""

from pathlib import Path
import sys
import numpy as np
from copy import deepcopy

from flowline.sweep import FlowlineSweep
from flowline.spinup import FlowlineSpinup
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry
import flowline.geometry as geometry


def main():
    # Parse command line arguments
    args = parse_sweep_cli_args("Run a spinup sweep example with melt factor sweep.")

    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / "output")

    # --- Base Geometry ---
    x_gr, zb_gr, w_geom = geometry.create_uniform_slope(
        bed_characteristic_length=10000,
        domain_extent=12000,
        x_gr_points=61,
        width=1000,
        elevation_drop=1000,
    )

    scale = 100
    length = 5000
    h_init = np.maximum(0, scale * (1 - x_gr / length))

    base_geometry = FlowlineGeometry(
        x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, h0=h_init
    )

    # --- Base Config and Forcing ---
    base_config = FlowlineConfig(
        ts=0, tf=500, delx=25, delt=0.00078125, deltout=1.0, min_thick=1.0
    )
    base_forcing = TemperaturePrecipitationForcing(ts=0, tf=500, P0=2.0, T0=8.0, mu=0.6)

    # --- Create FlowlineSpinup Objects ---
    # Each mu value gets its own spinup to reach steady state
    mu_values = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]
    spinup_objects = {}

    for i, mu in enumerate(mu_values):
        run_id = f"run_{i:04d}"
        spinup_forcing = deepcopy(base_forcing)
        spinup_forcing.mu = mu
        spinup_objects[run_id] = FlowlineSpinup(
            config=deepcopy(base_config), geometry=base_geometry, forcing=spinup_forcing
        )

    # --- Experimental Perturbations ---
    # Apply +0.2°C warming from the spun-up state to test response
    experimental_perturbations = {
        run_id: {"forcing.T0": lambda T0: T0 + 0.2} for run_id in spinup_objects
    }

    print("Spinup sweep setup:")
    print(f"  Melt factor values: {mu_values}")
    print("  Spinup duration: 500 years")
    print("  Perturbation: +0.2 deg C warming")
    print(f"  Total runs: {len(spinup_objects)}")

    # --- Run the Sweep ---
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        spinup_objects=spinup_objects,
        experimental_perturbations=experimental_perturbations,
        **get_sweep_cli_kwargs(args),
    )

    sweep.run()

    # --- Custom Post-processing ---
    print(f"\nSpinup sweep completed. Results saved to: {args.output_dir}")

    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"

    if combined_results_path.exists():
        import xarray as xr
        import matplotlib.pyplot as plt

        print("Creating custom analysis...")
        ds = xr.open_dataset(combined_results_path)

        fig, axes = plt.subplot_mosaic(
            [["length", "volume"], ["final_length", "final_volume"]], figsize=(12, 10)
        )
        fig.suptitle("Glacier Response to Melt Factor Sensitivity", fontsize=16)

        (ds["edge"] / 1000).plot.line(x="time", ax=axes["length"])
        axes["length"].set_title("Length Evolution")
        axes["length"].set_xlabel("Time (years)")
        axes["length"].set_ylabel("Length (km)")
        axes["length"].grid(True, alpha=0.3)

        ice_volume_km3 = (ds["h"] * ds["w"] * ds.attrs["delx"]).sum(dim="x") / 1e9
        ice_volume_km3.plot.line(x="time", ax=axes["volume"])
        axes["volume"].set_title("Volume Evolution")
        axes["volume"].set_xlabel("Time (years)")
        axes["volume"].set_ylabel("Volume (km3)")
        axes["volume"].grid(True, alpha=0.3)

        final_length_km = ds["edge"].isel(time=-1) / 1000
        final_volume_km3 = ice_volume_km3.isel(time=-1)

        axes["final_length"].plot(range(len(mu_values)), final_length_km, "o-")
        axes["final_length"].set_title("Final Length vs Melt Factor")
        axes["final_length"].set_xlabel("Run index")
        axes["final_length"].set_ylabel("Final Length (km)")
        axes["final_length"].grid(True, alpha=0.3)

        axes["final_volume"].plot(
            range(len(mu_values)), final_volume_km3, "o-", color="orange"
        )
        axes["final_volume"].set_title("Final Volume vs Melt Factor")
        axes["final_volume"].set_xlabel("Run index")
        axes["final_volume"].set_ylabel("Final Volume (km3)")
        axes["final_volume"].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "spinup_sweep_analysis.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Analysis plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
