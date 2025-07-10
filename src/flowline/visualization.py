import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import xarray as xr

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

    # Check if the run resulted in NaNs and create an error plot if so
    if ds.h.isel(time=-1).isnull().all():
        title = f"Run QC: {output_path.name.replace('.png', '')}"
        fig.suptitle(title, fontsize=14)
        axes[0].text(0.5, 0.5, 'Simulation Failed: Produced NaN values', 
                     ha='center', va='center', color='red', fontsize=12, transform=axes[0].transAxes)
        axes[0].axis('off')
        axes[1].axis('off')
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

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

    title = f"Run QC: {output_path.name.replace('.png', '')}"
    if 'run_parameters' in ds.attrs:
        try:
            run_params = json.loads(ds.attrs['run_parameters'])
            params_to_show = run_params.get('forcing', {})
            params_str = ', '.join([f'{k}={v}' for k, v in params_to_show.items()])
            if params_str:
                title += f" (params: {params_str})"
        except (json.JSONDecodeError, TypeError):
            # If run_parameters is not a valid JSON string, just use the default title.
            pass

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
    sweep_dims = [dim for dim in ds.coords if dim not in ['time', 'x']]

    if sweep_dims:
        df_edge = edge_plot.to_dataframe(name='length_km').reset_index()
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        if len(sweep_dims) > 2:
            print(f"Warning: More than 2 sweep dimensions. Plotting with color for '{hue_dim}' and style for '{style_dim}'.")

        sns.lineplot(
            data=df_edge, x='time', y='length_km', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend='auto'
        )
        if ax.get_legend():
            # Adjust figure to make space for legend and move it
            fig.tight_layout(rect=[0, 0, 0.8, 1])
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Fallback for single run
        edge_plot.plot(ax=ax, alpha=0.7)

    ax.set_title('Glacier Length Trajectories')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Length (km)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_length.png', dpi=150, bbox_inches='tight')
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

    if sweep_dims:
        df_vol = volume_plot.to_dataframe(name='volume_km3').reset_index()
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_vol, x='time', y='volume_km3', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend='auto'
        )
        if ax.get_legend():
            # Adjust figure to make space for legend and move it
            fig.tight_layout(rect=[0, 0, 0.8, 1])
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Fallback for single run
        volume_plot.plot(ax=ax, alpha=0.7)

    ax.set_title('Glacier Volume Trajectories')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Volume (km^3)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_volume.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Cumulative fractional length change
    fig, ax = plt.subplots(figsize=(10, 6))
    edge_initial = ds.edge.isel(time=0)
    edge_change = ds.edge - edge_initial
    total_edge_change = ds.edge.isel(time=-1) - edge_initial
    frac_edge_change = xr.where(total_edge_change != 0, edge_change / total_edge_change, 0)

    if sweep_dims:
        df_edge_change = frac_edge_change.to_dataframe(name='frac_length_change').reset_index()
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_edge_change, x='time', y='frac_length_change', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend='auto'
        )
        if ax.get_legend():
            # Adjust figure to make space for legend and move it
            fig.tight_layout(rect=[0, 0, 0.8, 1])
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Fallback for single run
        frac_edge_change.plot(ax=ax, alpha=0.7)

    ax.set_title('Cumulative Fractional Length Change')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Fractional Change')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_frac_length_change.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 4: Cumulative fractional volume change
    fig, ax = plt.subplots(figsize=(10, 6))
    volume_initial = volume.isel(time=0)
    volume_change = volume - volume_initial
    total_volume_change = volume.isel(time=-1) - volume_initial
    frac_volume_change = xr.where(total_volume_change != 0, volume_change / total_volume_change, 0)

    if sweep_dims:
        df_vol_change = frac_volume_change.to_dataframe(name='frac_volume_change').reset_index()
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_vol_change, x='time', y='frac_volume_change', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend='auto'
        )
        if ax.get_legend():
            # Adjust figure to make space for legend and move it
            fig.tight_layout(rect=[0, 0, 0.8, 1])
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Fallback for single run
        frac_volume_change.plot(ax=ax, alpha=0.7)

    ax.set_title('Cumulative Fractional Volume Change')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Fractional Change')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'sweep_qc_frac_volume_change.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Sweep QC plots saved in: {output_dir}")
