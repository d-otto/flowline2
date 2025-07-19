"""
run.py

Compares glacier response to warming on flat vs. convex bed geometries using target matching.
Analyzes changes in ELA, length, and volume between the two bed types.

This script demonstrates:
1. Creating flat, convex, and concave bed geometries  
2. Using target matching to ensure comparable initial glacier lengths across bed types
3. Running warming scenarios with lapse rate variations
4. Calculating ELA, length, and volume changes using package functions
5. Comparing glacier sensitivity between bed types

The target matching system optimizes temperature to achieve a consistent glacier length
across different bed geometries, ensuring fair comparison of sensitivity to warming.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray as xr

# Add src directory to path to allow direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.flowline.sweep import FlowlineSweep
from src.flowline.cli.utils import parse_sweep_cli_args
from src.flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from src.flowline.geometry import FlowlineGeometry, create_uniform_slope
from src.flowline.spinup import FlowlineSpinup, LengthOnlyCost, VolumeChangeRateDetector
from src.flowline.diagnostics import calc_ela
from src.flowline.visualization import plot_glacier_profile, plot_fractional_volume_timeseries, plot_volume_length_timeseries
from scipy.interpolate import splrep, splev

def create_matched_convex_bed(x_gr_flat, zb_gr_flat, convexity_amplitude=200):
    """Create convex bed profile that matches flat bed start and end elevations"""
    # Start and end elevations from flat bed
    z_start = zb_gr_flat[0]
    z_end = zb_gr_flat[-1]
    
    # Create convex shape using a parabolic perturbation
    x_normalized = (x_gr_flat - x_gr_flat[0]) / (x_gr_flat[-1] - x_gr_flat[0])
    
    # Parabolic perturbation that's zero at endpoints
    perturbation = convexity_amplitude * x_normalized * (1 - x_normalized)
    
    # Create convex bed by adding perturbation to linear interpolation
    zb_convex = z_start + (z_end - z_start) * x_normalized + perturbation
    
    # Use cubic spline to smooth the profile
    tck = splrep(x_gr_flat, zb_convex, s=0)
    zb_gr_convex = splev(x_gr_flat, tck)
    
    # Ensure exact match at endpoints
    zb_gr_convex[0] = z_start
    zb_gr_convex[-1] = z_end
    
    return zb_gr_convex

def analyze_glacier_metrics(dataset, temperature, base_forcing_params):
    """Extract glacier metrics from simulation results"""
    metrics = {}
    
    # Get final state
    final_h = dataset.h.isel(time=-1)
    
    # Calculate volume (integrate over glacier domain)
    glacier_mask = final_h > 0.01  # 1cm threshold
    if 'w' in dataset and 'delx' in dataset:
        volume = float((final_h * dataset.w * dataset.delx).where(glacier_mask).sum())
    else:
        # Fallback calculation - use constant width and delx
        delx = 50.0  # From grid setup
        w = 1000.0  # From base geometry
        volume = float((final_h * w * delx).where(glacier_mask).sum())
    
    # Get length - handle if edge is a scalar or array
    edge_final = dataset.edge.isel(time=-1)
    if edge_final.size == 1:
        length = float(edge_final)
    else:
        length = float(edge_final.values)
    
    # Calculate ELA using the package function
    ela = calc_ela(
        P0=base_forcing_params['P0'] * 1000,  # Convert to mm
        T0=temperature,
        gamma=base_forcing_params['gamma'] * 1000,  # Convert to °C/km
        mu=base_forcing_params['mu']
    )
    
    metrics['volume'] = volume
    metrics['length'] = length  
    metrics['ela'] = ela
    
    return metrics

def main():
    # Parse CLI arguments
    args = parse_sweep_cli_args("Compare glacier response to lapse rate variations on flat vs convex beds")
    
    # --- 1. Define Output Directory ---
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Example outputs will be saved to: {output_dir}")

    # --- 2. Create Configuration Objects ---
    # Base configuration
    base_config = FlowlineConfig(
        ts=0, 
        tf=500, 
        delt=0.0125/8, 
        delx=50, 
        deltout=1.0
    )
    
    
    # --- 3. Create Bed Geometries ---
    print("Creating bed geometries...")
    
    # Flat bed (uniform slope)
    x_gr_flat, zb_gr_flat, w_geom_flat = create_uniform_slope(
        domain_extent=15000,
        x_gr_points=51,
        elevation_drop=1000,
        bed_characteristic_length=8000,
        width=1000
    )
    h_init = np.maximum(0, 150 * (1 - x_gr_flat / 4000))  # Initial ice thickness
    flat_geometry = FlowlineGeometry(x_gr_flat, zb_gr_flat, w_geom_flat, 
                                   x_init=x_gr_flat, h_init=h_init)
    
    # Convex bed (matching flat bed endpoints)
    zb_gr_convex = create_matched_convex_bed(x_gr_flat, zb_gr_flat, convexity_amplitude=500)
    convex_geometry = FlowlineGeometry(x_gr_flat, zb_gr_convex, w_geom_flat,
                                     x_init=x_gr_flat, h_init=h_init)
    
    # Concave bed (matching flat bed endpoints)
    zb_gr_concave = create_matched_convex_bed(x_gr_flat, zb_gr_flat, convexity_amplitude=-500)
    concave_geometry = FlowlineGeometry(x_gr_flat, zb_gr_concave, w_geom_flat,
                                      x_init=x_gr_flat, h_init=h_init)
    
    # --- 4. Define Forcing Objects ---
    # Baseline temperature and experimental temperature
    baseline_temp = 7.0  # Used for spinup
    experimental_temp = 7.25  # Step change for experiment
    
    
    # Base forcing for experimental runs (warmed temperature)
    base_forcing = TemperaturePrecipitationForcing(
        T0=experimental_temp, P0=2.0, 
        gamma=6.5e-3, mu=0.65, 
        ts=base_config.ts, tf=base_config.tf
    )
    
    # --- 5. Set up Target Matching ---
    print("Setting up target matching...")
    
    # Lapse rate scenarios (5.0 and 7.0 K/km)
    lapse_rates = [5.0e-3, 7.0e-3]  # Convert to K/m
    
    # Target glacier length for consistent comparison
    target_length = 8000  # 8 km target length
    
    # Create FlowlineSpinup objects for each bed geometry and lapse rate combination
    def create_spinup_object(geometry, gamma_val, run_id_suffix):
        """Create FlowlineSpinup object with target matching for specific geometry and lapse rate"""
        
        # Create INDEPENDENT spinup config for this specific run
        spinup_config_custom = FlowlineConfig(
            ts=0,
            tf=1000,  # Longer time for steady state
            delt=0.0125/8,
            delx=50,
            deltout=1.0
        )
        
        # Create INDEPENDENT spinup forcing with this lapse rate
        spinup_forcing_custom = TemperaturePrecipitationForcing(
            T0=baseline_temp, P0=2.0, 
            gamma=gamma_val, mu=0.65, 
            ts=spinup_config_custom.ts, tf=spinup_config_custom.tf
        )
        
        # Configure target matching to achieve consistent glacier length
        target_matching = {
            'targets': {
                'target_length': target_length,
            },
            'adjustment_parameter': 'T0',
            'cost_function': LengthOnlyCost,
            'steady_state_detector': VolumeChangeRateDetector,
            'tolerance': 0.1,    # 0.1°C temperature tolerance
            'parameter_bounds': (5.0, 10.0),  # Wider temperature range
            'max_iterations': 100,  # More iterations
            'max_simulation_time': 1000  # Longer time for steady state
        }
        
        return FlowlineSpinup(
            config=spinup_config_custom,  # Use independent config
            geometry=geometry,
            forcing=spinup_forcing_custom,
            target_matching=target_matching
        )
    
    # Create spinup objects for each bed geometry and lapse rate combination
    bed_types = {'flat': flat_geometry, 'convex': convex_geometry, 'concave': concave_geometry}
    results = {}
    
    for bed_name, geometry in bed_types.items():
        print(f"Running {bed_name} bed simulations...")
        bed_output_dir = output_dir / f"{bed_name}_bed"
        
        # Create spinup objects for each lapse rate
        spinup_objects = {}
        experimental_perturbations = {}
        
        for i, gamma_val in enumerate(lapse_rates):
            run_id = f"run_{i:04d}"
            
            # Create spinup object with target matching for this geometry and lapse rate
            spinup_obj = create_spinup_object(geometry, gamma_val, run_id)
            spinup_objects[run_id] = spinup_obj
            
            # Experimental perturbation: temperature step change
            experimental_perturbations[run_id] = {
                'forcing.T0': lambda T0, temp_change=experimental_temp-baseline_temp: T0 + temp_change
            }
        
        # Run sweep with target-matched spinups
        sweep = FlowlineSweep(
            base_config=base_config,
            base_geometry=geometry,
            base_forcing=base_forcing,
            sweep_parameters={},  # No additional parameter sweep - controlled by spinup objects
            spinup_objects=spinup_objects,
            experimental_perturbations=experimental_perturbations,
            output_dir=str(bed_output_dir),
            workers=args.workers if hasattr(args, 'workers') else 4
        )
        sweep.run()
        results[bed_name] = bed_output_dir / "combined_results.nc"
    
    # --- 7. Post-processing and Analysis ---
    print("Analyzing results...")
    
    analysis = {}
    
    # Base forcing parameters for ELA calculation
    base_forcing_params = {
        'P0': base_forcing.P0,
        'gamma': base_forcing.gamma,
        'mu': base_forcing.mu
    }
    
    # Load and analyze each scenario
    for bed_type in ['flat', 'convex', 'concave']:
        analysis[bed_type] = {}
        
        # Get the combined results file
        combined_file = results[bed_type]
        if combined_file.exists():
            ds = xr.open_dataset(combined_file)
            
            for i, gamma in enumerate(lapse_rates):
                # Select data for this run (runs are numbered sequentially)
                run_id = f'run_{i:04d}'
                if 'run_id' in ds.dims:
                    gamma_data = ds.sel(run_id=run_id)
                else:
                    # If no run_id dimension, try to get by index
                    gamma_data = ds.isel(run_id=i) if 'run_id' in ds.dims else ds
                
                # Update base forcing params for this lapse rate
                current_forcing_params = base_forcing_params.copy()
                current_forcing_params['gamma'] = gamma
                
                # Extract metrics using experimental temperature
                metrics = analyze_glacier_metrics(gamma_data, temperature=experimental_temp, base_forcing_params=current_forcing_params)
                scenario_name = f'gamma_{gamma*1000:.1f}'
                analysis[bed_type][scenario_name] = metrics
        else:
            print(f"Warning: Could not find results file for {bed_type} bed: {combined_file}")
    
    # --- 8. Calculate Changes ---
    print("Calculating lapse rate response...")
    
    comparison = {}
    for bed_type in ['flat', 'convex', 'concave']:
        low_gamma_key = f'gamma_{lapse_rates[0]*1000:.1f}'
        high_gamma_key = f'gamma_{lapse_rates[1]*1000:.1f}'
        
        low_gamma = analysis[bed_type][low_gamma_key]
        high_gamma = analysis[bed_type][high_gamma_key]
        
        comparison[bed_type] = {
            'length_change': high_gamma['length'] - low_gamma['length'],
            'length_change_pct': ((high_gamma['length'] - low_gamma['length']) / low_gamma['length']) * 100,
            'volume_change': high_gamma['volume'] - low_gamma['volume'],
            'volume_change_pct': ((high_gamma['volume'] - low_gamma['volume']) / low_gamma['volume']) * 100,
            'ela_change': high_gamma['ela'] - low_gamma['ela'],
            'low_gamma_length': low_gamma['length'],
            'low_gamma_volume': low_gamma['volume'],
            'low_gamma_ela': low_gamma['ela']
        }
    
    # --- 9. Create Visualization ---
    print("Creating comparison plots...")
    
    # Load thickness data for visualization
    flat_ds = xr.open_dataset(results['flat'])
    convex_ds = xr.open_dataset(results['convex'])
    concave_ds = xr.open_dataset(results['concave'])
    
    # Create subplot mosaic layout - 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    mosaic = [
        ['flat_low', 'flat_high'],
        ['convex_low', 'convex_high']
    ]
    axes = fig.subplot_mosaic(mosaic)
    fig.suptitle(f'Glacier Response to Lapse Rate Variations: {baseline_temp}°C → {experimental_temp}°C Step Change', fontsize=16)
    
    # Helper function to plot each scenario
    def plot_scenario(ax, ds, bed_type, gamma_idx, gamma_name):
        # Get model grid and bed from dataset
        x_model = ds.x.values
        run_id = f'run_{gamma_idx:04d}'
        
        # Get bed profile
        if 'run_id' in ds.dims:
            zb_model = ds.zb.sel(run_id=run_id).values
        else:
            zb_model = ds.zb.isel(run_id=gamma_idx).values if 'run_id' in ds.dims else ds.zb.values
        
        # Get thickness data for initial and final timesteps
        if 'run_id' in ds.dims:
            scenario_data = ds.sel(run_id=run_id)
            h_initial = scenario_data.h.isel(time=0).values
            h_final = scenario_data.h.isel(time=-1).values
        else:
            # Fallback - try to get by index
            scenario_data = ds.isel(run_id=gamma_idx) if 'run_id' in ds.dims else ds
            h_initial = scenario_data.h.isel(time=0).values
            h_final = scenario_data.h.isel(time=-1).values
        
        # Get width and delx for area histogram
        if 'run_id' in ds.dims:
            w = ds.w.sel(run_id=run_id).values
        else:
            w = ds.w.isel(run_id=gamma_idx).values if 'run_id' in ds.dims else ds.w.values
        delx = ds.attrs.get('delx', 50)
        
        # Get ELA values
        gamma_val = lapse_rates[gamma_idx]
        ela_initial = calc_ela(P0=2000, T0=experimental_temp, gamma=gamma_val*1000, mu=0.65)
        ela_final = ela_initial  # Same since it's a step change, not gradual
        
        # Plot using the new function
        plot_glacier_profile(
            ax, x_model, zb_model, h_initial, h_final,
            ela_initial, ela_final,
            w=w, delx=delx,
            initial_label='Initial',
            final_label='Final',
            show_area_histogram=False
        )
        ax.set_title(f'{bed_type.title()} Bed - {gamma_name} Lapse Rate')
    
    # Plot all four scenarios
    plot_scenario(axes['flat_low'], flat_ds, 'flat', 0, 'Low (5.0 K/km)')
    plot_scenario(axes['flat_high'], flat_ds, 'flat', 1, 'High (7.0 K/km)')
    plot_scenario(axes['convex_low'], convex_ds, 'convex', 0, 'Low (5.0 K/km)')
    plot_scenario(axes['convex_high'], convex_ds, 'convex', 1, 'High (7.0 K/km)')
    
    plt.tight_layout()
    plot_path = output_dir / "bed_lapse_rate_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    
    # Create fractional volume timeseries plot using the new flexible function
    timeseries_path = output_dir / "fractional_volume_timeseries.png"
    plot_fractional_volume_timeseries(flat_ds, convex_ds, concave_ds, 
                                    labels=['Flat Bed', 'Convex Bed', 'Concave Bed'], 
                                    save_path=timeseries_path)
    print(f"Fractional volume timeseries plot saved to {timeseries_path}")
    
    # Create volume and length timeseries plot using the new flexible function
    vol_length_path = output_dir / "volume_length_timeseries.png"
    plot_volume_length_timeseries(flat_ds, convex_ds, concave_ds,
                                labels=['Flat Bed', 'Convex Bed', 'Concave Bed'], 
                                save_path=vol_length_path)
    print(f"Volume/length timeseries plot saved to {vol_length_path}")
    
    flat_ds.close()
    convex_ds.close()
    concave_ds.close()
    
    # --- 9. Print Results Summary ---
    print("\\n" + "="*60)
    print("GLACIER LAPSE RATE SENSITIVITY COMPARISON")
    print("="*60)
    print(f"Scenario: {baseline_temp}°C → {experimental_temp}°C step change with different lapse rates")
    print()
    
    for bed_type in ['flat', 'convex', 'concave']:
        print(f"{bed_type.upper()} BED:")
        c = comparison[bed_type]
        print(f"  Length change ({lapse_rates[0]*1000:.1f}→{lapse_rates[1]*1000:.1f} K/km): {c['length_change']/1000:.2f} km ({c['length_change_pct']:.1f}%)")
        print(f"  Volume change ({lapse_rates[0]*1000:.1f}→{lapse_rates[1]*1000:.1f} K/km): {c['volume_change']/1e9:.2f} km³ ({c['volume_change_pct']:.1f}%)")
        print(f"  ELA change ({lapse_rates[0]*1000:.1f}→{lapse_rates[1]*1000:.1f} K/km): {c['ela_change']:.0f} m")
        print()
    print("="*60)
    print(f"Configuration: Lapse rate sensitivity with {baseline_temp}°C → {experimental_temp}°C step change")
    print(f"Spinup temperature: {baseline_temp}°C")
    print(f"Experimental temperature: {experimental_temp}°C")
    print(f"Simulation time: {base_config.tf} years")
    print(f"Spatial resolution: {base_config.delx} m")
    print("="*60)

if __name__ == "__main__":
    main()