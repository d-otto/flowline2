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


def plot_fractional_volume_timeseries(flat_ds, convex_ds, lapse_rates, concave_ds=None, save_path=None):
    """
    Plot fractional volume change timeseries for different bed types and lapse rates.
    
    Parameters
    ----------
    flat_ds : xr.Dataset
        Dataset from flat bed simulations
    convex_ds : xr.Dataset
        Dataset from convex bed simulations
    lapse_rates : list
        List of lapse rate values (in K/m)
    concave_ds : xr.Dataset, optional
        Dataset from concave bed simulations
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Determine number of panels based on available datasets
    datasets = [('flat', flat_ds), ('convex', convex_ds)]
    if concave_ds is not None:
        datasets.append(('concave', concave_ds))
    
    # Create subplot layout
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    
    # Color scheme for lapse rates
    colors = plt.cm.viridis(np.linspace(0, 1, len(lapse_rates)))
    
    # Helper function to calculate and plot fractional volume change
    def plot_bed_type(ax, ds, bed_name):
        for i, gamma in enumerate(lapse_rates):
            # Select data for this lapse rate
            if 'forcing_gamma' in ds.dims:
                gamma_data = ds.sel(forcing_gamma=gamma)
            else:
                # Fallback if no dimension
                gamma_data = ds
            
            # Calculate volume (h * w * delx)
            if 'w' in gamma_data and 'delx' in gamma_data.attrs:
                volume = (gamma_data.h * gamma_data.w * gamma_data.attrs['delx']).sum(dim='x')
            else:
                # Fallback calculation
                delx = gamma_data.attrs.get('delx', 50)
                w = gamma_data.w if 'w' in gamma_data else 1000  # Default width
                volume = (gamma_data.h * w * delx).sum(dim='x')
            
            # Calculate fractional change (0 to 1: initial to final)
            initial_volume = volume.isel(time=0)
            final_volume = volume.isel(time=-1)
            fractional_volume = (volume - initial_volume) / (final_volume - initial_volume)
            
            # Plot timeseries
            gamma_label = f'{gamma*1000:.1f} K/km'
            ax.plot(gamma_data.time, fractional_volume, 
                   color=colors[i], linewidth=2, label=gamma_label)
        
        # Formatting
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Fractional Volume Change\n(V(t)-V₀)/(V_final-V₀)')
        ax.set_title(f'{bed_name.title()} Bed')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)  # Range from 0 to 1
    
    # Plot all bed types
    for i, (bed_name, ds) in enumerate(datasets):
        plot_bed_type(axes[i], ds, bed_name)
    
    # Overall title
    fig.suptitle('Fractional Volume Change Timeseries by Lapse Rate', fontsize=16)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_volume_length_timeseries(flat_ds, convex_ds, lapse_rates, concave_ds=None, save_path=None):
    """
    Plot volume and length timeseries for different bed types and lapse rates.
    
    Parameters
    ----------
    flat_ds : xr.Dataset
        Dataset from flat bed simulations
    convex_ds : xr.Dataset
        Dataset from convex bed simulations
    lapse_rates : list
        List of lapse rate values (in K/m)
    concave_ds : xr.Dataset, optional
        Dataset from concave bed simulations
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create 2x2 subplot grid: lapse rates as columns, variables as rows
    fig, axes = plt.subplots(2, len(lapse_rates), figsize=(7*len(lapse_rates), 10))
    
    # Ensure axes is 2D even for single lapse rate
    if len(lapse_rates) == 1:
        axes = axes.reshape(2, 1)
    
    # Determine available datasets and colors
    datasets = [('flat', flat_ds), ('convex', convex_ds)]
    bed_colors = {'flat': 'tab:blue', 'convex': 'tab:orange'}
    
    if concave_ds is not None:
        datasets.append(('concave', concave_ds))
        bed_colors['concave'] = 'tab:green'
    
    # Helper function to plot timeseries for a given lapse rate and variable
    def plot_variable(ax, gamma_val, variable_name):
        
        for bed_name, ds in datasets:
            # Select data for this lapse rate
            if 'forcing_gamma' in ds.dims:
                gamma_data = ds.sel(forcing_gamma=gamma_val)
            else:
                # Fallback if no dimension
                gamma_data = ds
            
            if variable_name == 'volume':
                # Calculate volume (h * w * delx) and convert to km³
                if 'w' in gamma_data and 'delx' in gamma_data.attrs:
                    values = (gamma_data.h * gamma_data.w * gamma_data.attrs['delx']).sum(dim='x') / 1e9
                else:
                    # Fallback calculation
                    delx = gamma_data.attrs.get('delx', 50)
                    w = gamma_data.w if 'w' in gamma_data else 1000  # Default width
                    values = (gamma_data.h * w * delx).sum(dim='x') / 1e9
                ylabel = 'Volume (km³)'
            
            elif variable_name == 'length':
                # Use edge data and convert to km
                values = gamma_data.edge / 1000
                ylabel = 'Length (km)'
            
            # Plot timeseries
            bed_label = f'{bed_name.title()} bed'
            ax.plot(gamma_data.time, values, 
                   color=bed_colors[bed_name], linewidth=2, label=bed_label)
        
        # Formatting
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(ylabel)
        gamma_label = f'{gamma_val*1000:.1f} K/km'
        ax.set_title(f'{variable_name.title()} - {gamma_label} Lapse Rate')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot all combinations
    variables = ['volume', 'length']
    
    for col, gamma in enumerate(lapse_rates):
        for row, variable in enumerate(variables):
            plot_variable(axes[row, col], gamma, variable)
    
    # Overall title
    fig.suptitle('Volume and Length Timeseries by Lapse Rate', fontsize=16)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
