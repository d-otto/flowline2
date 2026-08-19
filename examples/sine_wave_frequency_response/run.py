#!/usr/bin/env python3
"""
Sine Wave Bed Frequency Response Example

This example demonstrates how glaciers on different sine wave bed geometries
(varying frequencies) respond to warming, using target matching to ensure
comparable initial states.

The experiment:
1. Creates sine wave beds with different frequencies (1-5 wavelengths)
2. Uses target matching to spin up all glaciers to the same length (8km)
3. Applies uniform +0.5°C warming and compares response patterns
4. Analyzes how bed wavelength affects retreat sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
import sys
import xarray as xr
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from flowline.sweep import FlowlineSweep
from flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_function_profile
from flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector


def create_sine_wave_geometries(frequencies):
    """
    Create sine wave bed geometries with different frequencies.
    
    Parameters:
    -----------
    frequencies : list
        List of frequencies (number of wavelengths across domain)
    
    Returns:
    --------
    dict : Dictionary of geometries keyed by frequency
    """
    geometries = {}
    
    # Domain and geometric parameters
    domain_extent = 16000  # 16 km domain
    x_gr_points = 1000     # High-resolution points for geometry
    base_slope = 0.08      # 8% baseline grade (gentler slope)
    amplitude = 50        # 50m amplitude oscillations (smaller for stability)
    width = 1000          # 1000m constant width
    
    for freq in frequencies:
        # Create cosine wave elevation function (bed slopes downward from head to terminus)
        # Using cosine ensures the bed starts descending from x=0
        elevation_function = f"(base_slope * (domain_extent - x)) + amplitude * cos(frequency * 2 * pi * x / domain_extent) + 1000"
        
        function_kwargs = {
            'base_slope': base_slope,
            'amplitude': amplitude, 
            'frequency': freq,
            'domain_extent': domain_extent,
            'pi': np.pi
        }
        
        # Create geometry arrays
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=domain_extent,
            x_gr_points=x_gr_points,
            elevation_function=elevation_function,
            width=width,
            function_kwargs=function_kwargs
        )
        
        # Create simple initial ice thickness profile (thin wedge from terminus)
        # Set all glaciers to start at 8km so they begin at same terrain phase
        h_init = np.maximum(0, 100 * (1 - x_gr / 8000))  # Simple wedge shape, 8km extent
        
        # Create FlowlineGeometry object
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=h_init)
        geometries[freq] = geometry
        
        print(f"Created sine wave bed geometry with frequency {freq} wavelengths")
        print(f"  Domain: {domain_extent/1000:.1f} km, Elevation range: {zb_gr.min():.0f}-{zb_gr.max():.0f} m")
        
    return geometries


def create_spinup_objects(geometries, target_length=8000):
    """
    Create FlowlineSpinup objects with target matching for each geometry.
    
    Parameters:
    -----------
    geometries : dict
        Dictionary of FlowlineGeometry objects keyed by frequency
    target_length : float
        Target glacier length for all geometries (meters)
    
    Returns:
    --------
    dict : Dictionary of FlowlineSpinup objects
    """
    spinup_objects = {}
    
    # Spinup simulation parameters
    spinup_config = FlowlineConfig(
        ts=0,           # Start time
        tf=2000,        # End time (1000 years should be enough for steady state)
        delx=50,        # 50m grid spacing
        deltout=1,     # Output every 10 years during spinup
        delt=0.0125/16  # Smaller time step for stability
    )
    
    # Base climate forcing (will be adjusted by optimization)
    base_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=spinup_config.tf,
        T0=20.0,         # Initial guess for temperature (will be optimized)
        P0=2.0,         # 2 m/yr precipitation
        mu=0.6,        
        gamma=5e-3    # Temperature lapse rate (°C/m)
    )
    
    # Target matching configuration
    target_matching = {
        'targets': {
            'target_length': target_length,
        },
        'adjustment_parameter': 'T0',  # Optimize temperature (not 'forcing.T0')
        'cost_function': LengthOnlyCost,
        'steady_state_detector': VolumeChangeRateDetector,
        'max_simulation_time': spinup_config.tf,  # Maximum simulation time
        'bounds': (12.75, 13.5),  # Temperature bounds based on target ELA at bed elevations
        'optimization_options': {
           'maxiter': 40,
           'xatol': 0.005,            # Parameter tolerance
           'fatol': 10,            # Function tolerance (cost units)
           'adaptive': True,         # Adaptive parameters for better performance
           
        }
    }
    
    # Create spinup object for each frequency
    for freq, geometry in geometries.items():
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')  # freq_0p5 for 0.5
        
        # Create forcing with frequency-dependent initial guess  
        T0_initial = 20
        forcing = TemperaturePrecipitationForcing(
            ts=0,
            tf=spinup_config.tf,
            T0=T0_initial,
            P0=base_forcing.P0,
            mu=base_forcing.mu,
            gamma=base_forcing.gamma
        )
        
        # Create spinup object
        spinup_obj = FlowlineSpinup(
            config=spinup_config,
            geometry=geometry,
            forcing=forcing,
            target_matching=target_matching
        )
        
        spinup_objects[run_id] = spinup_obj
        print(f"Created spinup object for frequency {freq} with initial T0={forcing.T0:.1f}°C")
    
    return spinup_objects


def create_analysis_plots(output_dir, frequencies):
    """
    Create comprehensive analysis plots for the sine wave frequency response.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory containing results
    frequencies : list  
        List of frequencies used in the simulation
    """
    
    # Load combined results
    results_file = output_dir / "combined_results.nc"
    if not results_file.exists():
        print(f"Results file {results_file} not found, skipping analysis")
        return
        
    ds = xr.open_dataset(results_file)
    print(ds)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplot mosaic layout
    mosaic = [
        ['bed_geom', 'bed_geom', 'spinup_opt', 'spinup_opt'],
        ['initial_profiles', 'initial_profiles', 'response_ts', 'response_ts'], 
        ['final_profiles', 'final_profiles', 'sensitivity', 'sensitivity']
    ]
    
    axes = fig.subplot_mosaic(mosaic)
    
    # Colors for different frequencies
    colors = plt.cm.Set1(np.linspace(0, 1, len(frequencies)))
    
    # Plot 1: Bed geometry comparison
    ax = axes['bed_geom']
    for i, freq in enumerate(frequencies):
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')
        if run_id in ds.run_id.values:
            x = ds.x.values / 1000  # Convert to km
            zb = ds.zb.sel(run_id=run_id).values
            ax.plot(x, zb, color=colors[i], linewidth=2, label=f'{freq} wavelength{"s" if freq > 1 else ""}')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Bed Elevation (m)')
    ax.set_title('Sine Wave Bed Geometries')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Spinup optimization convergence
    ax = axes['spinup_opt']
    spinup_dir = output_dir / "spinup_profiles"
    if spinup_dir.exists():
        for i, freq in enumerate(frequencies):
            # Handle fractional frequencies in run_id
            if freq == int(freq):
                run_id = f"freq_{int(freq):d}"
            else:
                run_id = f"freq_{freq:g}".replace('.', 'p')
            spinup_file = spinup_dir / f"spinup_{run_id}.nc"
            if spinup_file.exists():
                spinup_ds = xr.open_dataset(spinup_file)
                # Plot length evolution during spinup
                if 'L' in spinup_ds.variables:
                    time_years = spinup_ds.time.values
                    length_km = spinup_ds.L.values / 1000
                    ax.plot(time_years, length_km, color=colors[i], linewidth=2, label=f'Freq {freq}')
    
    ax.axhline(y=8.0, color='black', linestyle='--', alpha=0.5, label='Target (8 km)')
    ax.set_xlabel('Spinup Time (years)')
    ax.set_ylabel('Glacier Length (km)')
    ax.set_title('Target Matching Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Initial ice thickness profiles (post-spinup)
    ax = axes['initial_profiles']
    for i, freq in enumerate(frequencies):
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')
        if run_id in ds.run_id.values:
            x = ds.x.values / 1000
            h_initial = ds.h.sel(run_id=run_id, time=0).values
            ax.plot(x, h_initial, color=colors[i], linewidth=2, label=f'Freq {freq}')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Ice Thickness (m)')
    ax.set_title('Initial Ice Profiles (Post-Spinup)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Response time series (length evolution)
    ax = axes['response_ts']
    for i, freq in enumerate(frequencies):
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')
        if run_id in ds.run_id.values and 'L' in ds.variables:
            time_years = ds.time.values
            length_km = ds.L.sel(run_id=run_id).values / 1000
            ax.plot(time_years, length_km, color=colors[i], linewidth=2, label=f'Freq {freq}')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Glacier Length (km)')
    ax.set_title('Response to +0.5°C Warming')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Final ice thickness profiles
    ax = axes['final_profiles']
    for i, freq in enumerate(frequencies):
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')
        if run_id in ds.run_id.values:
            x = ds.x.values / 1000
            h_final = ds.h.sel(run_id=run_id).isel(time=-1).values
            ax.plot(x, h_final, color=colors[i], linewidth=2, label=f'Freq {freq}')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Ice Thickness (m)')
    ax.set_title('Final Ice Profiles (After Warming)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Response sensitivity analysis
    ax = axes['sensitivity']
    if 'L' in ds.variables:
        initial_lengths = []
        final_lengths = []
        retreat_amounts = []
        
        for freq in frequencies:
            # Handle fractional frequencies in run_id
            if freq == int(freq):
                run_id = f"freq_{int(freq):d}"
            else:
                run_id = f"freq_{freq:g}".replace('.', 'p')
            if run_id in ds.run_id.values:
                L_initial = ds.L.sel(run_id=run_id, time=0).values / 1000
                L_final = ds.L.sel(run_id=run_id, time=-1).values / 1000
                retreat = L_initial - L_final
                
                initial_lengths.append(L_initial)
                final_lengths.append(L_final)
                retreat_amounts.append(retreat)
        
        ax.bar(frequencies, retreat_amounts, color=colors[:len(frequencies)], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Bed Frequency (wavelengths)')
        ax.set_ylabel('Retreat Amount (km)')
        ax.set_title('Retreat Sensitivity vs Bed Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add retreat values as text
        for i, retreat in enumerate(retreat_amounts):
            ax.text(frequencies[i], retreat + 0.1, f'{retreat:.2f} km', 
                   ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "sine_wave_frequency_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comprehensive analysis plot: {output_file}")


def main():
    """Main execution function."""
    
    # Parse command line arguments
    args = parse_sweep_cli_args("Sine Wave Bed Frequency Response Example")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "output")
    
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Clear output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        output_dir.mkdir()
    
    print("=== Sine Wave Bed Frequency Response Example ===")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    
    # Define sine wave frequencies (number of wavelengths across domain)
    frequencies = [0.5, 1, 2, 4, 8, 16, 32]  # Half, 1, 2, 4, 8 complete wavelengths
    print(f"Analyzing frequencies: {frequencies} wavelengths")
    
    # Create cost function visualization FIRST for debugging
    print("\n1. Creating cost function visualization for debugging...")
    cost_function = LengthOnlyCost()
    targets = {'target_length': 8000}  # Same target as used in optimization
    cost_curve_file = output_dir / "cost_function_curve.png"
    cost_function.plot_cost_curve(
        targets=targets, 
        domain_length=16000,  # Same as sine wave domain
        delx=50,              # Same as grid spacing
        output_file=str(cost_curve_file)
    )
    print(f"Cost function curve saved to: {cost_curve_file}")
    
    # Create sine wave bed geometries
    print("\n2. Creating sine wave bed geometries...")
    geometries = create_sine_wave_geometries(frequencies)
    
    # Create spinup objects with target matching
    print("\n3. Setting up target matching spinup...")
    spinup_objects = create_spinup_objects(geometries, target_length=8000)
    
    # Set up experimental perturbations (+0.5°C warming)
    print("\n4. Configuring warming perturbations...")
    experimental_perturbations = {}
    for freq in frequencies:
        # Handle fractional frequencies in run_id
        if freq == int(freq):
            run_id = f"freq_{int(freq):d}"
        else:
            run_id = f"freq_{freq:g}".replace('.', 'p')
        experimental_perturbations[run_id] = {
            'forcing.T0': lambda T0: T0 + 0.5  # +0.5°C warming
        }
    print(f"Applied +0.5°C warming to {len(experimental_perturbations)} scenarios")
    
    # Response simulation configuration
    response_config = FlowlineConfig(
        ts=0,           # Start time
        tf=200,         # 200 years response time
        delx=50,        # 50m grid spacing 
        deltout=1,      # Output every year
        delt=0.0125/32  # Smaller time step for stability
    )
    
    # Base response forcing (will be perturbed)
    response_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=200,
        T0=8.0,         # Will be overridden by spinup + perturbation
        P0=2.0,         # 2 m/yr precipitation
        mu=0.5,
        gamma=6.5e-3    # Temperature lapse rate (°C/m)
    )
    
    # Use a representative geometry for the base (will be overridden by spinup)
    base_geometry = geometries[2]  # Use frequency 2 as representative
    
    # Run the sweep
    print("\n5. Executing sweep with target matching and warming...")
    sweep = FlowlineSweep(
        base_config=response_config,
        base_geometry=base_geometry,
        base_forcing=response_forcing,
        sweep_parameters={},  # No additional parameter sweeps
        spinup_objects=spinup_objects,
        experimental_perturbations=experimental_perturbations,
        **get_sweep_cli_kwargs(args)
    )
    
    sweep.run()
    print("Sweep execution completed!")
    
    # Create analysis plots
    print("\n6. Creating analysis plots...")
    create_analysis_plots(output_dir, frequencies)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    print("Key outputs:")
    print(f"  - combined_results.nc: Main simulation results")
    print(f"  - spinup_profiles/: Spinup optimization results")
    print(f"  - cost_function_curve.png: Cost function visualization")
    print(f"  - sine_wave_frequency_analysis.png: Comprehensive analysis")
    print("\nThis example demonstrates:")
    print("  • Target matching creates comparable initial states")
    print("  • Symmetric exponential cost function for better optimization")
    print("  • Different bed frequencies show varying retreat sensitivity")
    print("  • Bed geometry wavelength affects glacier response to warming")


if __name__ == "__main__":
    main()