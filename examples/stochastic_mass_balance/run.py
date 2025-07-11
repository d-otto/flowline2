#!/usr/bin/env python3
"""
Stochastic temperature forcing example demonstrating the impact of temperature noise variability.

This example shows how to:
1. Generate white noise timeseries with different temperature standard deviations using numpy RNG
2. Apply stochastic temperature forcing with varying noise levels
3. Analyze the resulting glacier length and volume distributions
4. Create comprehensive plots showing sensitivity to temperature noise amplitude

This demonstrates the full power of Python configuration for complex parameter generation.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep
from src.flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from src.flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from src.flowline.geometry import FlowlineGeometry
import src.flowline.geometry as geometry_module

def generate_temperature_noise(ts, tf, std_dev, rng):
    """
    Generate white noise timeseries for temperature with specified standard deviation.
    
    Parameters
    ----------
    ts, tf : float
        Start and end time (in years)
    std_dev : float
        Standard deviation of temperature noise (°C)
    rng : numpy.random.Generator
        Random number generator for reproducibility
    
    Returns
    -------
    temp_noise : array
        Temperature white noise with mean=0 and specified std_dev, one value per year
    """
    # Generate one noise value per year (year_idx goes from 0 to tf-ts-1)
    n_years = int(tf - ts)
    # Generate standard normal noise, then scale to desired std_dev
    noise_timeseries = rng.normal(loc=0, scale=std_dev, size=n_years)
    return noise_timeseries

def main():
    # Parse command line arguments
    args = parse_sweep_cli_args("Run stochastic mass balance sweep with varying noise levels.")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent / 'output')
    
    # --- Set up reproducible random number generation ---
    base_seed = 42
    rng = np.random.default_rng(base_seed)
    
    # --- Base Configuration ---
    base_config = FlowlineConfig(
        ts=0,
        tf=500,
        delx=25,
        delt=0.0125/16,  # Stable timestep
        deltout=1.0,  # Output every year
        min_thick=10.0
    )
    
    # --- Base Geometry ---
    # Use consistent geometry for all runs
    x_gr, zb_gr, w_geom = geometry_module.create_uniform_slope(
        bed_characteristic_length=10000,
        domain_extent=12000,
        x_gr_points=61,
        width=1000,
        elevation_drop=1000
    )
    
    # Create initial ice thickness profile  
    scale = 100
    length = 6000
    h_init = np.maximum(0, scale * (1 - x_gr / length))
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr,
        w_geom=w_geom,
        x_init=x_gr,
        h_init=h_init
    )
    
    # --- Generate Stochastic Temperature Timeseries ---
    # Different temperature noise standard deviations to test
    temp_noise_std_devs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # °C
    base_temperature = 8.0  # °C mean temperature T0
    
    # Generate temperature noise timeseries for each standard deviation
    temp_noise_dict = {}
    for std_dev in temp_noise_std_devs:
        temp_noise = generate_temperature_noise(
            base_config.ts, base_config.tf, std_dev, rng
        )
        temp_noise_dict[f'std_{std_dev}'] = temp_noise
    
    print(f"Generated temperature noise timeseries for std devs: {temp_noise_std_devs} °C")
    print(f"Base temperature T0: {base_temperature} °C")
    print(f"Simulation length: {base_config.tf} years")
    print(f"Time points per series: {len(next(iter(temp_noise_dict.values())))}")
    
    # --- Set up Parameter Sweep with Stochastic Temperature Forcing ---
    # Generate all the temperature noise timeseries we'll need
    # Need to generate for the longer of spinup duration or main run duration
    spinup_duration = 200  # From spinup_config
    max_duration = max(base_config.tf, spinup_duration)
    
    temp_noise_timeseries = []
    for i, std_dev in enumerate(temp_noise_std_devs):
        # Generate the actual temperature noise timeseries for this run
        local_rng = np.random.default_rng(base_seed + i)  # Different seed for each run
        temp_noise = generate_temperature_noise(
            base_config.ts, max_duration, std_dev, local_rng
        )
        temp_noise_timeseries.append(temp_noise)
    
    # Create base forcing object with TemperaturePrecipitationForcing
    # For spinup, we want no temperature noise (Tp=0)
    base_forcing = TemperaturePrecipitationForcing(
        T0=base_temperature,  # Base temperature
        P0=2.0,              # Precipitation (m/yr)
        gamma=0.0065,        # Temperature lapse rate (6.5 °C/km)
        mu=0.65,             # Melt factor
        ts=base_config.ts,
        tf=max_duration,
        T=np.zeros(max_duration),  # No temperature noise initially
        P=np.zeros(max_duration)   # No precipitation noise
    )
    
    # Set up parameter sweep over the different temperature noise timeseries
    sweep_parameters = {
        'forcing.Tp': temp_noise_timeseries  # Sweep over temperature perturbation (Tp)
    }
    
    print(f"Sweep will test temperature noise standard deviations: {temp_noise_std_devs} °C")
    print(f"Total runs: {len(temp_noise_timeseries)}")
    print(f"Temperature noise timeseries lengths: {[len(ts) for ts in temp_noise_timeseries]}")
    
    # --- Spinup Configuration ---
    # Enable spinup with no temperature noise for steady initialization
    spinup_config = {
        'enabled': True,
        'config': {
            'tf': spinup_duration
        }
    }
    
    # --- Run the Sweep ---
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        spinup_config=spinup_config,
        **get_sweep_cli_kwargs(args)
    )
    
    sweep.run()
    
    # --- Custom Post-processing ---
    print(f"\nStochastic mass balance sweep completed. Results saved to: {args.output_dir}")
    
    # Load and analyze results
    output_dir = Path(args.output_dir)
    combined_results_path = output_dir / "combined_results.nc"
    
    if combined_results_path.exists():
        import xarray as xr
        
        print("Creating comprehensive stochastic analysis plots...")
        ds = xr.open_dataset(combined_results_path)
        
        print("Dataset dimensions:", ds.dims)
        print("Dataset variables:", list(ds.variables.keys()))
        
        # Check if we have any data to plot
        if 'edge' not in ds.variables:
            print("No edge data found in combined results. Check for failed runs.")
            return
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Stochastic Mass Balance Analysis', fontsize=16)
        
        # Calculate derived quantities
        final_lengths_km = ds['edge'].isel(time=-1) / 1000  # Convert to km
        ice_volume_km3 = (ds['h'] * ds['w'] * ds.attrs['delx']).sum(dim='x') / 1e9
        final_volumes_km3 = ice_volume_km3.isel(time=-1)
        
        # Calculate glacier area for specific mass balance
        glacier_area_km2 = ((ds['h'] > 0.1) * ds['w'] * ds.attrs['delx']).sum(dim='x') / 1e6
        
        # 1. Length time series (overlapping)
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        # The sweep created a dimension based on the parameter name, check what we have
        sweep_dim = None
        for dim in ds.dims:
            if 'forcing' in dim:
                sweep_dim = dim
                break
        
        if sweep_dim and len(ds[sweep_dim]) >= len(temp_noise_std_devs):
            for i, (std_dev, color) in enumerate(zip(temp_noise_std_devs, colors)):
                length_series = ds['edge'].isel({sweep_dim: i}) / 1000
                label = f'σ_T = {std_dev:.2f} °C'
                axes[0, 0].plot(ds['time'], length_series, alpha=0.8, linewidth=2, 
                               color=color, label=label)
        else:
            length_series = ds['edge'] / 1000
            if len(length_series.shape) > 1:
                # If we have multiple runs, plot them all
                for i in range(length_series.shape[0]):
                    axes[0, 0].plot(ds['time'], length_series[i], alpha=0.8, linewidth=2, 
                                   color=colors[i % len(colors)], label=f'Run {i}')
            else:
                axes[0, 0].plot(ds['time'], length_series, alpha=0.8, linewidth=2)
        
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('Glacier Length (km)')
        axes[0, 0].set_title('Length Evolution (All Noise Levels)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Volume time series (overlapping)
        if sweep_dim and len(ds[sweep_dim]) >= len(temp_noise_std_devs):
            for i, (std_dev, color) in enumerate(zip(temp_noise_std_devs, colors)):
                volume_series = ice_volume_km3.isel({sweep_dim: i})
                label = f'σ_T = {std_dev:.2f} °C'
                axes[0, 1].plot(ds['time'], volume_series, alpha=0.8, linewidth=2, 
                               color=color, label=label)
        else:
            if len(ice_volume_km3.shape) > 1:
                for i in range(ice_volume_km3.shape[0]):
                    axes[0, 1].plot(ds['time'], ice_volume_km3[i], alpha=0.8, linewidth=2, 
                                   color=colors[i % len(colors)], label=f'Run {i}')
            else:
                axes[0, 1].plot(ds['time'], ice_volume_km3, alpha=0.8, linewidth=2)
        
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('Glacier Volume (km³)')
        axes[0, 1].set_title('Volume Evolution (All Noise Levels)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Total mass balance time series 
        time_points = np.arange(base_config.ts, base_config.tf)  # One point per year
        if 'total_mass_balance' in ds.variables and sweep_dim and len(ds[sweep_dim]) >= len(temp_noise_std_devs):
            for i, (std_dev, color) in enumerate(zip(temp_noise_std_devs, colors)):
                mb_series = ds['total_mass_balance'].isel({sweep_dim: i})
                label = f'σ_T = {std_dev:.2f} °C'
                # Only plot first few years to avoid overcrowding
                plot_years = min(100, len(mb_series))
                axes[1, 0].plot(ds['time'][:plot_years], mb_series[:plot_years], alpha=0.7, linewidth=1, 
                               color=color, label=label)
        
        axes[1, 0].set_xlabel('Time (years)')
        axes[1, 0].set_ylabel('Total Mass Balance (m/yr)')
        axes[1, 0].set_title('Total Mass Balance Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Temperature anomaly (Tp) time series
        time_points = np.arange(base_config.ts, base_config.tf)  # One point per year
        for i, (std_dev, color) in enumerate(zip(temp_noise_std_devs, colors)):
            # Regenerate the temperature noise for display purposes (same seed)
            local_rng = np.random.default_rng(base_seed + i)
            temp_noise = generate_temperature_noise(
                base_config.ts, base_config.tf, std_dev, local_rng
            )
            label = f'σ_T = {std_dev:.2f} °C'
            # Only plot first few years to avoid overcrowding
            plot_years = min(100, len(temp_noise))
            axes[1, 1].plot(time_points[:plot_years], temp_noise[:plot_years], alpha=0.8, linewidth=1, 
                           color=color, label=label)
        
        axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.8, label='Mean: 0 °C')
        axes[1, 1].set_xlabel('Time (years)')
        axes[1, 1].set_ylabel('Temperature Anomaly (°C)')
        axes[1, 1].set_title('Temperature Anomaly (Tp) Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # 5. Histogram of length timeseries
        all_lengths_km = ds['edge'] / 1000  # All timesteps, convert to km
        
        if sweep_dim and len(ds[sweep_dim]) > 1:
            # Create overlapping histograms for each noise level
            for i, std_dev in enumerate(temp_noise_std_devs[:len(ds[sweep_dim])]):
                length_timeseries = all_lengths_km.isel({sweep_dim: i})
                axes[2, 0].hist(length_timeseries.values, bins=20, alpha=0.4, 
                               label=f'σ_T = {std_dev:.2f}°C', color=colors[i % len(colors)],
                               edgecolor=colors[i % len(colors)], linewidth=1.5)
            axes[2, 0].legend()
        else:
            # Single histogram of all length values across time
            axes[2, 0].hist(all_lengths_km.values.flatten(), bins=30, 
                           alpha=0.7, edgecolor='black', color='skyblue')
        
        axes[2, 0].set_xlabel('Glacier Length (km)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Distribution of Length Over Time')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Histogram of volume timeseries
        all_volumes_km3 = ice_volume_km3  # All timesteps
        
        if sweep_dim and len(ds[sweep_dim]) > 1:
            # Create overlapping histograms for each noise level
            for i, std_dev in enumerate(temp_noise_std_devs[:len(ds[sweep_dim])]):
                volume_timeseries = all_volumes_km3.isel({sweep_dim: i})
                axes[2, 1].hist(volume_timeseries.values, bins=20, alpha=0.4, 
                               label=f'σ_T = {std_dev:.2f}°C', color=colors[i % len(colors)],
                               edgecolor=colors[i % len(colors)], linewidth=1.5)
            axes[2, 1].legend()
        else:
            # Single histogram of all volume values across time
            axes[2, 1].hist(all_volumes_km3.values.flatten(), bins=30, 
                           alpha=0.7, edgecolor='black', color='lightcoral')
        
        axes[2, 1].set_xlabel('Glacier Volume (km³)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Distribution of Volume Over Time')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_plot_path = output_dir / "stochastic_analysis.png"
        plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to: {analysis_plot_path}")
        
        # Print summary statistics
        print(f"\nSummary statistics:")
        print(f"Noise standard deviations tested: {temp_noise_std_devs} m/yr")
        print(f"Base temeprature: {base_temperature} m/yr")
        
        if 'edge' in ds.variables:
            print(f"Final glacier lengths: {final_lengths_km.min().values:.1f} - {final_lengths_km.max().values:.1f} km")
            print(f"Final volumes: {final_volumes_km3.min().values:.1f} - {final_volumes_km3.max().values:.1f} km³")
            
            # Calculate sensitivity
            if len(final_lengths_km) > 1:
                length_sensitivity = (final_lengths_km.max() - final_lengths_km.min()) / (max(temp_noise_std_devs) - min(temp_noise_std_devs))
                volume_sensitivity = (final_volumes_km3.max() - final_volumes_km3.min()) / (max(temp_noise_std_devs) - min(temp_noise_std_devs))
                print(f"Length sensitivity: {length_sensitivity.values:.2f} km per m/yr noise")
                print(f"Volume sensitivity: {volume_sensitivity.values:.2f} km³ per m/yr noise")
    
    print("\nThis example demonstrates:")
    print("- Reproducible stochastic mass balance generation using numpy RNG")
    print("- Parameter sweeps over different noise amplitudes")
    print("- Comprehensive time series and distribution analysis")
    print("- Sensitivity analysis of glacier response to noise variability")
    print("- Advanced post-processing with overlapping time series plots")

if __name__ == "__main__":
    main()