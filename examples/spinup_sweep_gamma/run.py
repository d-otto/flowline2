#!/usr/bin/env python3
"""
Spinup sweep example using FlowlineSpinup objects.

Demonstrates a parameter sweep with a shared spinup, where all lapse rate
values start from the same steady-state profile before the main experimental run.
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
    args = parse_sweep_cli_args("Run a spinup sweep example with lapse rate sweep.")

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
        x_gr=x_gr, zb_gr=zb_gr, w_geom=w_geom, x_init=x_gr, h_init=h_init
    )

    # --- Base Config and Forcing ---
    base_config = FlowlineConfig(
        ts=0, tf=500, delx=25, delt=0.00078125, deltout=1.0, min_thick=1.0
    )
    base_forcing = TemperaturePrecipitationForcing(
        ts=0, tf=500, P0=2.0, T0=8.2, mu=0.65, gamma=0.0065
    )

    # --- Shared Spinup ---
    # All gamma runs share the same steady-state spinup since gamma only varies
    # in the main experimental runs, not during spinup.
    spinup_forcing = deepcopy(base_forcing)
    spinup_forcing.T0 = 8.0
    spinup_forcing.gamma = 0.0065

    shared_spinup = FlowlineSpinup(
        config=deepcopy(base_config),
        geometry=base_geometry,
        forcing=spinup_forcing,
    )

    # --- Sweep Parameters ---
    gamma_values = [0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]
    sweep_parameters = {"forcing.gamma": gamma_values}

    print("Spinup sweep setup:")
    print(f"  Lapse rate values: {gamma_values}")
    print(f"  Spinup climate: T0={spinup_forcing.T0}C, P0={spinup_forcing.P0}m/yr")
    print(f"  Main run climate: T0={base_forcing.T0}C, P0={base_forcing.P0}m/yr")
    print(f"  Total runs: {len(gamma_values)}")

    # --- Run the Sweep ---
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        spinup_objects=shared_spinup,
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
        fig.suptitle("Glacier Response to Temperature Lapse Rate", fontsize=16)

        (ds["edge"] / 1000).plot.line(x="time", hue="forcing_gamma", ax=axes["length"])
        axes["length"].set_title("Length Evolution")
        axes["length"].set_xlabel("Time (years)")
        axes["length"].set_ylabel("Length (km)")
        axes["length"].grid(True, alpha=0.3)

        ice_volume_km3 = (ds["h"] * ds["w"] * ds.attrs["delx"]).sum(dim="x") / 1e9
        ice_volume_km3.plot.line(x="time", hue="forcing_gamma", ax=axes["volume"])
        axes["volume"].set_title("Volume Evolution")
        axes["volume"].set_xlabel("Time (years)")
        axes["volume"].set_ylabel("Volume (km3)")
        axes["volume"].grid(True, alpha=0.3)

        final_length_km = ds["edge"].isel(time=-1) / 1000
        final_volume_km3 = ice_volume_km3.isel(time=-1)

        axes["final_length"].plot(ds["forcing_gamma"] * 1000, final_length_km, "o-")
        axes["final_length"].set_title("Final Length vs Lapse Rate")
        axes["final_length"].set_xlabel("Lapse Rate (deg C/km)")
        axes["final_length"].set_ylabel("Final Length (km)")
        axes["final_length"].grid(True, alpha=0.3)

        axes["final_volume"].plot(
            ds["forcing_gamma"] * 1000, final_volume_km3, "o-", color="orange"
        )
        axes["final_volume"].set_title("Final Volume vs Lapse Rate")
        axes["final_volume"].set_xlabel("Lapse Rate (deg C/km)")
        axes["final_volume"].set_ylabel("Final Volume (km3)")
        axes["final_volume"].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "spinup_sweep_gamma_analysis.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Analysis plot saved to: {plot_path}")

        print("\nSummary:")
        print(
            f"  Lapse rate range: {ds['forcing_gamma'].min().values * 1000:.1f}"
            f" - {ds['forcing_gamma'].max().values * 1000:.1f} deg C/km"
        )
        print(
            f"  Final length range: {final_length_km.min().values:.1f}"
            f" - {final_length_km.max().values:.1f} km"
        )
        print(
            f"  Final volume range: {final_volume_km3.min().values:.1f}"
            f" - {final_volume_km3.max().values:.1f} km3"
        )


if __name__ == "__main__":
    main()
