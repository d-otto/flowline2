import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def init_plot():
    """Initialize real-time plotting figure"""
    fig = plt.figure(figsize=(8, 12), dpi=100)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax = np.empty((3, 2), dtype='object')
    
    for i in range(3):
        for j in range(2):
            ax[i, j] = fig.add_subplot(gs[i, j])
    
    return fig, ax

import json

def rt_plot(model, t, i):
    """Update real-time plot"""
    # This would contain the real-time plotting logic
    # Move the _rt_plot method content from flowline2d.py
    pass


def plot_run_qc(ds, output_path):
    """
    Generate a QC plot for a single flowline run.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from a single model run.
    output_path : str or Path
        Path to save the plot figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Plot 1: Glacier length and volume over time
    ax = axes[0]
    # Length
    (ds.edge / 1e3).plot(ax=ax, label='Length')
    ax.set_ylabel('Length (km)')
    ax.set_xlabel('Time (years)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Volume
    ax2 = ax.twinx()
    volume = (ds.h * ds.w * ds.attrs['delx']).sum(dim='x')
    (volume / 1e9).plot(ax=ax2, color='C1', label='Volume')
    ax2.set_ylabel('Volume (km^3)')

    # Legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    ax.set_title('Glacier Evolution')

    # Plot 2: Final ice thickness profile
    ax = axes[1]
    final_h = ds.h.isel(time=-1)
    final_surface = ds.zb + final_h
    x_km = ds.x.values / 1e3

    ax.plot(x_km, final_surface, label='Ice Surface')
    ax.plot(x_km, ds.zb, color='k', label='Bed')
    ax.fill_between(x_km, ds.zb, final_surface, where=(final_h > 0), color='lightblue', alpha=0.7)

    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Elevation (m)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title(f'Final Profile at year {ds.time.values[-1]:.1f}')

    try:
        run_params = json.loads(ds.attrs['run_parameters'])
        title = f"Run QC: {output_path.name.replace('.png', '')}"
        params_to_show = run_params.get('forcing', {})
        params_str = ', '.join([f'{k}={v}' for k, v in params_to_show.items()])
        if params_str:
            title += f" (params: {params_str})"
    except (json.JSONDecodeError, KeyError):
        title = f"Run QC: {output_path.name.replace('.png', '')}"

    fig.suptitle(title, fontsize=14)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sweep_qc(ds, output_dir):
    """
    Generate QC plots for a flowline sweep.

    Parameters
    ----------
    ds : xr.Dataset
        The combined dataset from a sweep.
    output_dir : Path
        Directory to save plot figures.
    """
    if 'time' not in ds.dims:
        print("Warning: 'time' dimension not found in combined dataset. Skipping sweep QC plots.")
        return

    # Ensure dataset is loaded into memory to avoid issues with file handles
    ds = ds.load()

    # Plot 1: Glacier length trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    edge_plot = ds.edge / 1e3
    if edge_plot.ndim > 2:
        # Stack non-time dimensions to plot all runs
        non_time_dims = [dim for dim in edge_plot.dims if dim != 'time']
        edge_plot = edge_plot.stack(run=non_time_dims)

    if edge_plot.ndim > 1:
        edge_plot.plot.line(ax=ax, x='time', add_legend=False, alpha=0.7)
    else:
        edge_plot.plot(ax=ax, alpha=0.7)

    ax.set_title('Glacier Length Trajectories')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Length (km)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_length.png', dpi=150)
    plt.close(fig)

    # Plot 2: Glacier volume trajectories
    if 'delx' in ds.attrs:
        delx = ds.attrs['delx']
    elif 'config_delx' in ds.coords:
        delx = ds['config_delx']  # Use DataArray for broadcasting
    else:
        print("Warning: 'delx' not found. Using default of 50m for volume calculation.")
        delx = 50

    volume = (ds.h * ds.w * delx).sum(dim='x')
    fig, ax = plt.subplots(figsize=(10, 6))
    volume_plot = volume / 1e9
    if volume_plot.ndim > 2:
        # Stack non-time dimensions to plot all runs
        non_time_dims = [dim for dim in volume_plot.dims if dim != 'time']
        volume_plot = volume_plot.stack(run=non_time_dims)

    if volume_plot.ndim > 1:
        volume_plot.plot.line(ax=ax, x='time', add_legend=False, alpha=0.7)
    else:
        volume_plot.plot(ax=ax, alpha=0.7)

    ax.set_title('Glacier Volume Trajectories')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Volume (km^3)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_volume.png', dpi=150)
    plt.close(fig)

    print(f"Sweep QC plots saved in: {output_dir}")
