import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, Any, Optional, List, Dict, Union
import numpy as np
import seaborn as sns
import xarray as xr

def init_plot() -> Tuple[Any, np.ndarray]:
    """Initialize real-time plotting figure"""
    fig = plt.figure(figsize=(8, 12), dpi=100)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax = np.empty((3, 2), dtype='object')
    
    for i in range(3):
        for j in range(2):
            ax[i, j] = fig.add_subplot(gs[i, j])
    
    return fig, ax

import json

def _format_value_for_display(value: Any) -> str:
    """
    Format a value for display in titles and legends.
    
    Arrays and sequences are replaced with their shape information.
    Scalars are returned as-is.
    
    Parameters
    ----------
    value : any
        The value to format
        
    Returns
    -------
    str
        Formatted string representation
    """
    if isinstance(value, (list, tuple)):
        arr = np.array(value)
        return f"<array shape={arr.shape}>"
    elif isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    else:
        return str(value)

def _format_dataframe_for_plotting(df: Any, sweep_dims: List[str]) -> Any:
    """
    Format dataframe coordinates for plotting by replacing arrays with shape info.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    sweep_dims : list
        List of sweep dimension names
        
    Returns
    -------
    pd.DataFrame
        Formatted dataframe with readable coordinate values
    """
    df_formatted = df.copy()
    for dim in sweep_dims:
        if dim in df_formatted.columns:
            df_formatted[dim] = df_formatted[dim].apply(_format_value_for_display)
    return df_formatted

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
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), layout='constrained')

    # Plot 1: Glacier length and volume over time
    ax = axes[0]
    # Length
    edge_km = ds.edge / 1e3
    edge_km.plot(ax=ax, label='Length')
    ax.set_ylabel('Length (km)')
    ax.set_xlabel('Time (years)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Volume
    ax2 = ax.twinx()
    volume = (ds.h * ds.w * ds.attrs['delx']).sum(dim='x') / 1e9
    volume.plot(ax=ax2, color='C1', label='Volume')
    ax2.set_ylabel('Volume (km^3)')

    # Legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    ax.set_title('Glacier Evolution')

    # Plot 2: Total mass balance over time
    ax = axes[1]

    total_mb = (ds.F * ds.w * ds.attrs['delx']).sum(dim='x') / 1e9
    total_mb.plot(ax=ax)
    ax.set_ylabel('Flux (km^3/yr)')

    ax.set_xlabel('Time (years)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('Total Mass Balance')

    # Plot 3: Final ice thickness profile
    ax = axes[2]
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
    # if 'run_parameters' in ds.attrs:
    #     try:
    #         run_params = json.loads(ds.attrs['run_parameters'])
    #         param_items = []
    #         # Format all parameters, not just forcing parameters
    #         for section_name, section_params in run_params.items():
    #             if isinstance(section_params, dict):
    #                 for k, v in section_params.items():
    #                     formatted_value = _format_value_for_display(v)
    #                     param_items.append(f"{section_name}.{k}={formatted_value}")
    #             else:
    #                 # Handle top-level parameters
    #                 formatted_value = _format_value_for_display(section_params)
    #                 param_items.append(f"{section_name}={formatted_value}")
    #         params_str = ', '.join(param_items)
    #         if params_str:
    #             title += f" (params: {params_str})"
    #     except (json.JSONDecodeError, TypeError):
    #         # If run_parameters is not a valid JSON string, just use the default title.
    #         pass

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150)
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
        df_edge = _format_dataframe_for_plotting(df_edge, sweep_dims)
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        if len(sweep_dims) > 2:
            print(f"Warning: More than 2 sweep dimensions. Plotting with color for '{hue_dim}' and style for '{style_dim}'.")

        sns.lineplot(
            data=df_edge, x='time', y='length_km', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend=False
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
        df_vol = _format_dataframe_for_plotting(df_vol, sweep_dims)
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_vol, x='time', y='volume_km3', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend=False
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
        df_edge_change = _format_dataframe_for_plotting(df_edge_change, sweep_dims)
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_edge_change, x='time', y='frac_length_change', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend=False
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
        df_vol_change = _format_dataframe_for_plotting(df_vol_change, sweep_dims)
        hue_dim = sweep_dims[0]
        style_dim = sweep_dims[1] if len(sweep_dims) > 1 else None

        sns.lineplot(
            data=df_vol_change, x='time', y='frac_volume_change', hue=hue_dim,
            style=style_dim, ax=ax, palette='viridis', legend=False
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


def plot_glacier_profile(ax, x, zb, h_initial, h_final, ela_initial, ela_final, 
                        w=None, delx=None, initial_label="Initial", final_label="Final",
                        show_area_histogram=True, bin_size=100):
    """
    Plot glacier bed, initial/final thickness profiles, ELAs, and area histogram.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The primary axis to plot on
    x : array_like
        Spatial coordinates (m)
    zb : array_like
        Bed elevation (m)
    h_initial : array_like
        Initial ice thickness (m)
    h_final : array_like
        Final ice thickness (m)
    ela_initial : float
        Initial equilibrium line altitude (m)
    ela_final : float
        Final equilibrium line altitude (m)
    w : array_like, optional
        Glacier width (m). If provided with delx, enables area histogram.
    delx : float, optional
        Spatial resolution (m). If provided with w, enables area histogram.
    initial_label : str, optional
        Label for initial profile (default: "Initial")
    final_label : str, optional
        Label for final profile (default: "Final")
    show_area_histogram : bool, optional
        Whether to show area histogram on twin axis (default: True)
    bin_size : float, optional
        Elevation bin size for histogram in meters (default: 100)
        
    Returns
    -------
    ax_twin : matplotlib.axes.Axes or None
        The twin axis used for histogram, or None if not created
    """
    # Convert x to km for plotting
    x_km = x / 1000
    
    # Calculate surface elevations
    surface_initial = zb + h_initial
    surface_final = zb + h_final
    
    # Plot bed profile (black line)
    ax.plot(x_km, zb, 'k-', linewidth=2, label='Bed', zorder=3)
    
    # Plot initial profile (light blue with light shading)
    ax.plot(x_km, surface_initial, color='lightblue', linewidth=2, label=initial_label)
    ax.fill_between(x_km, zb, surface_initial, where=(h_initial > 0.01), 
                   edgecolor='lightblue', facecolor="none", alpha=0.3, hatch='///')
    
    # Plot final profile (light blue with hatching)
    ax.plot(x_km, surface_final, color='lightblue', linewidth=2, label=final_label)
    ax.fill_between(x_km, zb, surface_final, where=(h_final > 0.01), 
                   color='lightblue', alpha=0.7)
    
    # Add ELA lines
    ax.axhline(ela_initial, color='black', linestyle='--', alpha=0.7, 
              label=f'ELA {initial_label.lower()} ({ela_initial:.0f}m)')
    ax.axhline(ela_final, color='black', linestyle='-', alpha=0.7, 
              label=f'ELA {final_label.lower()} ({ela_final:.0f}m)')
    
    # Set labels and formatting
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Elevation (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add area histogram on twin axis if requested and data available
    ax_twin = None
    if show_area_histogram and w is not None and delx is not None:
        ax_twin = ax.twinx()
        
        # Calculate elevation range for binning
        min_elev = min(zb.min(), surface_initial.min(), surface_final.min())
        max_elev = max(zb.max(), surface_initial.max(), surface_final.max())
        
        # Create elevation bins
        elevation_bins = np.arange(
            np.floor(min_elev / bin_size) * bin_size,
            np.ceil(max_elev / bin_size) * bin_size + bin_size,
            bin_size
        )
        
        # Calculate area at each elevation (average of initial and final)
        areas = []
        bin_centers = []
        
        for i in range(len(elevation_bins) - 1):
            bin_bottom = elevation_bins[i]
            bin_top = elevation_bins[i + 1]
            bin_center = (bin_bottom + bin_top) / 2
            
            # Find points within this elevation bin (using average surface)
            avg_surface = (surface_initial + surface_final) / 2
            mask = (avg_surface >= bin_bottom) & (avg_surface < bin_top)
            
            if np.any(mask):
                # Calculate area (width * length where length is delx per grid point)
                area = np.sum(w[mask] * delx) / 1e6  # Convert to km²
                areas.append(area)
                bin_centers.append(bin_center)
        
        # Plot histogram
        if areas:
            ax_twin.barh(bin_centers, areas, height=bin_size*0.8, 
                        color='lightgrey', alpha=0.7, label='Area')
            ax_twin.set_xlabel('Area (km²)')
            ax_twin.set_ylabel('')
            ax_twin.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            
            # Add thin legend for area
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper right')
    
    return ax_twin


def plot_fractional_volume_timeseries(*datasets, labels=None, save_path=None):
    """
    Plot fractional volume change timeseries for different datasets.
    
    This function automatically detects the sweep dimension and iterates over all runs.
    
    Parameters
    ----------
    *datasets : xr.Dataset
        Variable number of datasets to plot (e.g., flat_ds, convex_ds, concave_ds)
    labels : list of str, optional
        Labels for each dataset. If None, uses 'Dataset 1', 'Dataset 2', etc.
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Extract datasets and create labels
    datasets = list(datasets)
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    
    # Create subplot layout
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    
    # Helper function to calculate and plot fractional volume change
    def plot_bed_type(ax, ds, bed_name):
        # Automatically detect sweep dimension
        sweep_dims = [dim for dim in ds.dims if dim not in ['x', 'time']]
        if not sweep_dims:
            # No sweep dimension - single run
            run_data = ds
            volume = calculate_volume(run_data)
            fractional_volume = calculate_fractional_change(volume)
            ax.plot(run_data.time, fractional_volume, linewidth=2, label='Single Run')
        else:
            # Use the first sweep dimension found
            sweep_dim = sweep_dims[0]
            n_runs = ds.sizes[sweep_dim]
            colors = plt.cm.viridis(np.linspace(0, 1, n_runs))
            
            for i in range(n_runs):
                run_data = ds.isel({sweep_dim: i})
                volume = calculate_volume(run_data)
                fractional_volume = calculate_fractional_change(volume)
                
                # Create label from coordinate if available
                if sweep_dim in ds.coords:
                    coord_value = ds.coords[sweep_dim].isel({sweep_dim: i}).values
                    if sweep_dim == 'forcing_gamma':
                        label = f'{coord_value*1000:.1f} K/km'
                    elif sweep_dim == 'run_id':
                        label = f'Run {i}'
                    else:
                        label = f'{sweep_dim}={coord_value}'
                else:
                    label = f'Run {i}'
                
                ax.plot(run_data.time, fractional_volume, 
                       color=colors[i], linewidth=2, label=label)
        
        # Formatting
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Fractional Volume Change\n(V(t)-V₀)/(V_final-V₀)')
        ax.set_title(f'{bed_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)  # Range from 0 to 1
    
    def calculate_volume(run_data):
        """Calculate volume from thickness data"""
        if 'w' in run_data and 'delx' in run_data.attrs:
            return (run_data.h * run_data.w * run_data.attrs['delx']).sum(dim='x')
        else:
            # Fallback calculation
            delx = run_data.attrs.get('delx', 50)
            w = run_data.w if 'w' in run_data else 1000  # Default width
            return (run_data.h * w * delx).sum(dim='x')
    
    def calculate_fractional_change(volume):
        """Calculate fractional change from initial to final"""
        initial_volume = volume.isel(time=0)
        final_volume = volume.isel(time=-1)
        return (volume - initial_volume) / (final_volume - initial_volume)
    
    # Plot all datasets
    for i, (ds, label) in enumerate(zip(datasets, labels)):
        plot_bed_type(axes[i], ds, label)
    
    # Overall title
    fig.suptitle('Fractional Volume Change Timeseries', fontsize=16)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_volume_length_timeseries(*datasets, labels=None, save_path=None):
    """
    Plot volume and length timeseries for different datasets.
    
    This function automatically detects the sweep dimension and creates a grid
    showing volume and length timeseries for each run in each dataset.
    
    Parameters
    ----------
    *datasets : xr.Dataset
        Variable number of datasets to plot
    labels : list of str, optional
        Labels for each dataset. If None, uses 'Dataset 1', 'Dataset 2', etc.
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Extract datasets and create labels
    datasets = list(datasets)
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    
    # Determine number of runs from first dataset
    first_ds = datasets[0]
    sweep_dims = [dim for dim in first_ds.dims if dim not in ['x', 'time']]
    n_runs = first_ds.sizes[sweep_dims[0]] if sweep_dims else 1
    
    # Create subplot grid: runs as columns, variables as rows
    fig, axes = plt.subplots(2, n_runs, figsize=(7*n_runs, 10))
    
    # Ensure axes is 2D even for single run
    if n_runs == 1:
        axes = axes.reshape(2, 1)
    
    # Color scheme for datasets
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    # Helper functions
    def calculate_volume(run_data):
        """Calculate volume from thickness data and convert to km³"""
        if 'w' in run_data and 'delx' in run_data.attrs:
            return (run_data.h * run_data.w * run_data.attrs['delx']).sum(dim='x') / 1e9
        else:
            delx = run_data.attrs.get('delx', 50)
            w = run_data.w if 'w' in run_data else 1000
            return (run_data.h * w * delx).sum(dim='x') / 1e9
    
    def calculate_length(run_data):
        """Calculate length from edge data and convert to km"""
        return run_data.edge / 1000
    
    def get_run_label(ds, run_idx):
        """Generate run label from dataset coordinates"""
        sweep_dims = [dim for dim in ds.dims if dim not in ['x', 'time']]
        if not sweep_dims:
            return 'Single Run'
        
        sweep_dim = sweep_dims[0]
        if sweep_dim in ds.coords:
            coord_value = ds.coords[sweep_dim].isel({sweep_dim: run_idx}).values
            if sweep_dim == 'forcing_gamma':
                return f'{coord_value*1000:.1f} K/km'
            elif sweep_dim == 'run_id':
                return f'Run {run_idx}'
            else:
                return f'{sweep_dim}={coord_value}'
        else:
            return f'Run {run_idx}'
    
    # Plot volume and length for each run
    variables = [('volume', 'Volume (km³)', calculate_volume), 
                 ('length', 'Length (km)', calculate_length)]
    
    for run_idx in range(n_runs):
        for var_idx, (var_name, ylabel, calc_func) in enumerate(variables):
            ax = axes[var_idx, run_idx]
            
            # Plot each dataset
            for ds_idx, (ds, ds_label) in enumerate(zip(datasets, labels)):
                color = colors[ds_idx % len(colors)]
                
                # Get data for this run
                sweep_dims = [dim for dim in ds.dims if dim not in ['x', 'time']]
                if sweep_dims:
                    run_data = ds.isel({sweep_dims[0]: run_idx})
                else:
                    run_data = ds
                
                # Calculate variable
                values = calc_func(run_data)
                
                # Plot timeseries
                ax.plot(run_data.time, values, 
                       color=color, linewidth=2, label=ds_label)
            
            # Formatting
            ax.set_xlabel('Time (years)')
            ax.set_ylabel(ylabel)
            
            # Title with run information
            run_label = get_run_label(datasets[0], run_idx)
            ax.set_title(f'{var_name.title()} - {run_label}')
            ax.grid(True, alpha=0.3)
            
            # Only show legend on first plot
            if var_idx == 0 and run_idx == 0:
                ax.legend()
    
    # Overall title
    fig.suptitle('Volume and Length Timeseries', fontsize=16)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
