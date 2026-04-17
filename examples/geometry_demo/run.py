#!/usr/bin/env python3
"""
Geometry Demo: Spline-based geometry creation and plotting

This example demonstrates the new spline-based geometry creation function
and the comprehensive geometry plotting capabilities added to FlowlineGeometry.

Features demonstrated:
- create_spline_profile() with complex bed elevation control points
- Spline-based width variation
- FlowlineGeometry.plot_geometry() method
- Comprehensive visualization of bed geometry, gradients, and initial profiles
"""

import os

from flowline.geometry import (
    create_spline_profile, create_uniform_slope, create_concave_profile,
    create_function_profile, FlowlineGeometry
)
import numpy as np

def create_spline_geometries():
    """Create multiple geometry examples using splines."""
    
    print("=== Creating Spline-Based Geometries ===\n")
    
    # Example 1: Complex overdeepened valley with variable width
    print("1. Complex overdeepened valley with spline-based width variation")
    
    bed_control_points = [
        (2000, 1800),   # High accumulation area
        (4000, 1200),   # Start of overdeepening  
        (6000, 800),    # Deepest point
        (8000, 600),    # Still overdeepened
        (12000, 200)    # Near terminus step
    ]
    
    width_control_points = [
        (0, 2500),      # Wide cirque
        (3000, 1800),   # Narrowing into valley
        (7000, 1200),   # Narrow through overdeepening
        (10000, 800),   # Constrained valley
        (15000, 400)    # Narrow terminus
    ]
    
    x_gr1, zb_gr1, w_geom1 = create_spline_profile(
        domain_extent=15000, 
        x_gr_points=150,
        z_start=2200, 
        z_end=0,
        control_points=bed_control_points,
        width_control_points=width_control_points,
        smoothing=0  # Exact interpolation
    )
    
    print(f"   Created {len(x_gr1)} points, elevation: {zb_gr1.min():.0f}-{zb_gr1.max():.0f} m")
    print(f"   Width variation: {w_geom1.min():.0f}-{w_geom1.max():.0f} m\n")
    
    # Example 2: Smooth exponential-like profile using smoothing
    print("2. Smooth exponential-like profile using spline smoothing")
    
    # Create control points that would normally give a jagged profile
    noisy_points = [(i*1000, 1800*np.exp(-i/8) + 50*np.sin(i)) 
                    for i in range(1, 15)]
    
    x_gr2, zb_gr2, w_geom2 = create_spline_profile(
        domain_extent=15000,
        x_gr_points=150,
        z_start=2000,
        z_end=50,
        control_points=noisy_points,
        width=1000,  # Constant width
        smoothing=500.0,  # Heavy smoothing
        spline_degree=3   # Cubic spline
    )
    
    print(f"   Smoothed profile: {len(x_gr2)} points")
    print(f"   Smooth elevation range: {zb_gr2.min():.0f}-{zb_gr2.max():.0f} m\n")
    
    # Example 3: Linear width transition
    print("3. Simple spline bed with linear width transition")
    
    x_gr3, zb_gr3, w_geom3 = create_spline_profile(
        domain_extent=10000,
        x_gr_points=100,
        z_start=1500,
        z_end=0,
        control_points=[(2500, 1100), (7500, 300)],  # Simple control points
        w_start=2000,   # Linear width variation
        w_end=500
    )
    
    print(f"   Simple profile: {len(x_gr3)} points")
    print(f"   Linear width: {w_geom3[0]:.0f} → {w_geom3[-1]:.0f} m\n")
    
    return [(x_gr1, zb_gr1, w_geom1, "Complex overdeepened valley"),
            (x_gr2, zb_gr2, w_geom2, "Smoothed exponential profile"), 
            (x_gr3, zb_gr3, w_geom3, "Linear width transition")]


def create_function_geometries():
    """Create multiple geometry examples using mathematical functions."""
    
    print("=== Creating Function-Based Geometries ===\n")
    
    # Example 1: Exponential decay bed with function-based width
    print("1. Exponential decay bed with function-based width variation")
    
    x_gr1, zb_gr1, w_geom1 = create_function_profile(
        domain_extent=12000,
        x_gr_points=120,
        elevation_function=lambda x: 2200 * np.exp(-x/6000),
        width_function=lambda x: 1800 - 800 * (x / 12000)**0.7
    )
    
    print(f"   Created {len(x_gr1)} points, elevation: {zb_gr1.min():.0f}-{zb_gr1.max():.0f} m")
    print(f"   Width variation: {w_geom1.min():.0f}-{w_geom1.max():.0f} m\n")
    
    # Example 2: Sinusoidal bed with parametric string expressions
    print("2. Sinusoidal bed profile with parametric string expressions")
    
    x_gr2, zb_gr2, w_geom2 = create_function_profile(
        domain_extent=10000,
        x_gr_points=100,
        elevation_function="base_slope * (domain_extent - x) + amplitude * sin(frequency * x / domain_extent)",
        width_function="min_width + width_range * exp(-decay * x / domain_extent)",
        function_kwargs={
            'base_slope': 0.18,    # 18% grade baseline
            'amplitude': 300,      # 300m amplitude oscillations
            'frequency': 3,        # 3 wavelengths across domain
            'min_width': 600,      # Minimum width
            'width_range': 1200,   # Width range
            'decay': 1.5          # Exponential width decay
        }
    )
    
    print(f"   Parametric profile: {len(x_gr2)} points")
    print(f"   Elevation range: {zb_gr2.min():.0f}-{zb_gr2.max():.0f} m")
    print(f"   Width range: {w_geom2.min():.0f}-{w_geom2.max():.0f} m\n")
    
    # Example 3: Power law bed with step width transition
    print("3. Power law bed profile with step width transition")
    
    # Define step function for width
    step_width_func = lambda x: np.where(x < 5000, 2000, np.where(x < 8000, 1200, 700))
    
    x_gr3, zb_gr3, w_geom3 = create_function_profile(
        domain_extent=10000,
        x_gr_points=100,
        elevation_function=lambda x: 1600 * (1 - (x/10000)**1.8),
        width_function=step_width_func
    )
    
    print(f"   Power law profile: {len(x_gr3)} points")
    print(f"   Elevation: {zb_gr3.min():.0f}-{zb_gr3.max():.0f} m")
    print(f"   Step widths: {w_geom3[0]:.0f} → {w_geom3[50]:.0f} → {w_geom3[-1]:.0f} m\n")
    
    # Example 4: Complex mathematical expression
    print("4. Complex mathematical bed with hyperbolic tangent transitions")
    
    x_gr4, zb_gr4, w_geom4 = create_function_profile(
        domain_extent=15000,
        x_gr_points=150,
        elevation_function="upper_plateau + (lower_plateau - upper_plateau) * (tanh((x - transition_point)/transition_width) + 1) / 2",
        w_start=2500, w_end=400,  # Linear width variation
        function_kwargs={
            'upper_plateau': 1800,     # High elevation plateau
            'lower_plateau': 200,      # Low elevation plateau
            'transition_point': 7500,  # Midpoint transition
            'transition_width': 1500   # Smoothness of transition
        }
    )
    
    print(f"   Complex transition: {len(x_gr4)} points")
    print(f"   Elevation plateaus: {zb_gr4.max():.0f} → {zb_gr4.min():.0f} m")
    print(f"   Linear width: {w_geom4[0]:.0f} → {w_geom4[-1]:.0f} m\n")
    
    print("Function-based advantages:")
    print("   ✓ Direct mathematical control over bed shape")
    print("   ✓ Parametric expressions for systematic studies")
    print("   ✓ Complex mathematical functions (trigonometric, exponential, hyperbolic)")
    print("   ✓ Easy integration with parameter sweeps and optimization")
    print("   ✓ Both lambda functions and string expressions supported\n")
    
    return [(x_gr1, zb_gr1, w_geom1, "Exponential decay with function width"),
            (x_gr2, zb_gr2, w_geom2, "Sinusoidal bed with parametric strings"),
            (x_gr3, zb_gr3, w_geom3, "Power law bed with step widths"),
            (x_gr4, zb_gr4, w_geom4, "Complex tanh transition")]


def compare_geometry_methods():
    """Compare traditional vs spline geometry creation methods."""
    
    print("=== Comparing Geometry Creation Methods ===\n")
    
    domain_extent = 10000
    x_gr_points = 100
    
    # Traditional methods
    print("Traditional geometry functions:")
    
    x_uniform, zb_uniform, w_uniform = create_uniform_slope(
        domain_extent=domain_extent, x_gr_points=x_gr_points,
        elevation_drop=1500, width=1000, 
        bed_characteristic_length=domain_extent
    )
    print(f"   Uniform slope: Linear from {zb_uniform[0]:.0f} to {zb_uniform[-1]:.0f} m")
    
    x_concave, zb_concave, w_concave = create_concave_profile(
        domain_extent=domain_extent, x_gr_points=x_gr_points,
        elevation_drop=1500, width=1000,
        bed_characteristic_length=domain_extent, perturbation=-200
    )
    print(f"   Concave profile: {zb_concave.min():.0f} to {zb_concave.max():.0f} m (overdeepened)")
    
    # Spline equivalent
    print("\nSpline-based equivalents:")
    
    x_spline_simple, zb_spline_simple, w_spline_simple = create_spline_profile(
        domain_extent=domain_extent, x_gr_points=x_gr_points,
        z_start=1500, z_end=0, width=1000
    )
    print(f"   Simple spline: {zb_spline_simple[0]:.0f} to {zb_spline_simple[-1]:.0f} m")
    
    x_spline_over, zb_spline_over, w_spline_over = create_spline_profile(
        domain_extent=domain_extent, x_gr_points=x_gr_points,
        z_start=1500, z_end=0, width=1000,
        control_points=[(5000, 1050)]  # Overdeepen at midpoint
    )
    print(f"   Spline overdeepened: {zb_spline_over.min():.0f} to {zb_spline_over.max():.0f} m")
    
    print("\nSpline advantages:")
    print("   ✓ Precise control through specified points")
    print("   ✓ Flexible width variations (constant, linear, or spline-based)")
    print("   ✓ Smoothing options for noisy control points") 
    print("   ✓ Access to all scipy spline parameters via kwargs\n")
    
    return [(x_uniform, zb_uniform, w_uniform, "Traditional uniform"),
            (x_spline_simple, zb_spline_simple, w_spline_simple, "Spline equivalent")]


def demonstrate_plotting():
    """Demonstrate comprehensive geometry plotting."""
    
    print("=== Demonstrating Geometry Plotting ===\n")
    
    # Create a complex geometry for plotting demo
    bed_points = [(2000, 1600), (5000, 1000), (8000, 400), (12000, 100)]
    width_points = [(0, 2000), (4000, 1500), (10000, 800), (15000, 500)]
    
    x_gr, zb_gr, w_geom = create_spline_profile(
        domain_extent=15000, x_gr_points=150,
        z_start=1800, z_end=0,
        control_points=bed_points,
        width_control_points=width_points
    )
    
    # Set up geometry object
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
    geometry.setup_grid(delx=50)
    
    # Add realistic initial ice profile
    # Thicker ice at higher elevations, tapering to zero
    ice_thickness = np.maximum(0, 
        200 * (geometry.zb / geometry.zb.max())**0.5 * 
        np.exp(-0.3 * (geometry.x / geometry.x.max())**2)
    )
    geometry.h0 = ice_thickness
    
    print(f"Created geometry: {geometry.nxs} grid points, {np.mean(np.diff(geometry.x)):.0f}m spacing")
    print(f"Ice thickness: {ice_thickness.max():.0f}m max, {np.sum(ice_thickness > 0)/len(ice_thickness)*100:.1f}% coverage")
    
    # Create output directory in the example folder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotting examples
    print("\nCreating geometry plots:")
    
    # 1. Basic plot (bed + width)
    print("   1. Basic geometry plot")
    fig1, axes1 = geometry.plot_geometry(figsize=(12, 6))
    fig1.savefig(f'{output_dir}/geometry_basic.png', dpi=150, bbox_inches='tight')
    
    # 2. With gradients
    print("   2. Geometry with gradients")  
    fig2, axes2 = geometry.plot_geometry(figsize=(12, 10), show_gradients=True)
    fig2.savefig(f'{output_dir}/geometry_with_gradients.png', dpi=150, bbox_inches='tight')
    
    # 3. With plan view
    print("   3. Geometry with plan view")
    fig3, axes3 = geometry.plot_geometry(
        figsize=(16, 8),
        show_plan_view=True
    )
    fig3.savefig(f'{output_dir}/geometry_with_plan_view.png', dpi=150, bbox_inches='tight')
    
    # 4. Comprehensive plot (all features)
    print("   4. Comprehensive plot (all features)")
    fig4, axes4 = geometry.plot_geometry(
        figsize=(18, 12), 
        show_gradients=True, 
        show_initial_profile=True,
        show_plan_view=True
    )
    fig4.savefig(f'{output_dir}/geometry_comprehensive.png', dpi=150, bbox_inches='tight')
    
    print(f"\nSaved plots to {output_dir}/")
    print("   - geometry_basic.png: Bed elevation and width")
    print("   - geometry_with_gradients.png: + bed slope and width gradients")
    print("   - geometry_with_plan_view.png: + plan view (top-down glacier outline)")
    print("   - geometry_comprehensive.png: + all features (gradients, initial profile, plan view)")
    
    # Close figures to avoid memory issues
    import matplotlib.pyplot as plt
    plt.close('all')
    
    return geometry


def plot_all_geometries(spline_examples, function_examples, comparison_examples):
    """Plot and save all geometry profiles created in the demo."""
    
    print("=== Plotting All Geometry Profiles ===\n")
    
    # Create output directory in the example folder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    all_examples = [
        ("spline", spline_examples),
        ("function", function_examples), 
        ("comparison", comparison_examples)
    ]
    
    plot_count = 0
    for category, examples in all_examples:
        print(f"Plotting {category} geometries:")
        
        for i, (x_gr, zb_gr, w_geom, description) in enumerate(examples):
            # Set up geometry object
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
            geometry.setup_grid(delx=50)
            
            # Add realistic initial ice profile based on bed elevation
            ice_thickness = np.maximum(0,
                150 * (geometry.zb / geometry.zb.max())**0.6 * 
                np.exp(-0.2 * (geometry.x / geometry.x.max())**1.5)
            )
            geometry.h0 = ice_thickness
            
            # Create comprehensive plot for each geometry
            filename_base = f"{category}_geometry_{i+1:02d}_{description.lower().replace(' ', '_')}"
            
            try:
                fig, axes = geometry.plot_geometry(
                    figsize=(16, 10),
                    show_gradients=True,
                    show_initial_profile=True,
                    show_plan_view=True
                )
                
                # Add descriptive title with proper spacing
                fig.suptitle(f"{category.title()} Geometry: {description}", 
                           fontsize=14, fontweight='bold', y=0.95)
                
                # Save plot
                fig.savefig(f'{output_dir}/{filename_base}.png', dpi=150, bbox_inches='tight')
                
                print(f"   {i+1}. {description} → {filename_base}.png")
                plot_count += 1
                
                # Close figure to avoid memory issues
                import matplotlib.pyplot as plt
                plt.close(fig)
                
            except Exception as e:
                print(f"   {i+1}. {description} → ERROR: {e}")
    
    print(f"\nSaved {plot_count} geometry plots to {output_dir}/")
    return plot_count


def main():
    """Run the complete geometry demonstration."""
    
    print("🏔️  Flowline Geometry Demo")
    print("=" * 50)
    print("Demonstrating spline-based and function-based geometry creation and plotting\n")
    
    # Create various spline geometries
    spline_examples = create_spline_geometries()
    
    # Create various function-based geometries
    function_examples = create_function_geometries()
    
    # Compare with traditional methods  
    comparison_examples = compare_geometry_methods()
    
    # Demonstrate plotting capabilities
    demo_geometry = demonstrate_plotting()
    
    # Plot and save all geometry profiles
    plot_count = plot_all_geometries(spline_examples, function_examples, comparison_examples)
    
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print("✓ Created multiple spline-based geometries with various features")
    print("✓ Created multiple function-based geometries with mathematical expressions")
    print("✓ Compared traditional vs spline vs function geometry creation methods")
    print("✓ Demonstrated comprehensive geometry plotting capabilities")
    print(f"✓ Generated {plot_count + 4} individual geometry plots with complete visualizations")
    
    print(f"\nKey features of create_spline_profile():")
    print("• Pass-through points: Exact control over bed elevation at specific locations")
    print("• Flexible width: Constant, linear transition, or spline-based variation") 
    print("• Smoothing options: From exact interpolation to heavily smoothed approximation")
    print("• Scipy integration: Direct access to all splrep() parameters via **kwargs")
    
    print(f"\nKey features of create_function_profile():")
    print("• Mathematical control: Direct function specification of bed elevation z(x)")
    print("• Multiple input types: Lambda functions, string expressions, or callable objects")
    print("• Function-based width: Both elevation and width can be specified as functions")
    print("• Parametric expressions: String expressions with customizable parameters")
    print("• Complex mathematics: Full numpy function library (trigonometric, exponential, hyperbolic)")
    print("• Parameter sweeps: Easy integration with systematic parameter studies")
    
    print(f"\nKey features of FlowlineGeometry.plot_geometry():")
    print("• Multi-panel visualization: Bed, width, gradients, initial profiles, plan view")
    print("• High-res vs model grid: Comparison of input geometry vs interpolated grid")
    print("• Plan view: Top-down glacier outline showing width variation and bed elevation")
    print("• Flexible options: Choose which elements to include in plots")
    print("• Professional styling: Proper labels, legends, and layout")
    
    return demo_geometry

if __name__ == "__main__":
    main()