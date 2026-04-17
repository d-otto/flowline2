import pytest
import numpy as np
import tempfile
import os
import dill

from flowline.geometry import (
    FlowlineGeometry, GeometryError, create_uniform_slope,
    create_concave_profile, create_convex_profile, create_variable_width,
    create_spline_profile, create_function_profile
)

@pytest.fixture
def basic_geometry_params():
    """Standard geometry parameters for testing."""
    domain_extent = 15000  # 15 km
    return {
        'bed_characteristic_length': domain_extent,
        'domain_extent': domain_extent,
        'x_gr_points': 41,
        'elevation_drop': 1000,  # m
        'width': 1000,  # m
    }

class TestGeometryHelpers:
    """Tests for standalone geometry creation functions."""

    def test_create_uniform_slope(self, basic_geometry_params):
        """Test uniform slope creation."""
        params = basic_geometry_params.copy()
        params.pop('bed_characteristic_length') # Not used by this one directly
        x_gr, zb_gr, w_geom = create_uniform_slope(
            bed_characteristic_length=basic_geometry_params['bed_characteristic_length'], **params)
        
        assert len(x_gr) == params['x_gr_points']
        assert zb_gr[0] == params['elevation_drop']
        assert zb_gr[-1] < 1.0 # Should be close to zero
        assert np.all(w_geom == params['width'])

    def test_create_concave_profile(self, basic_geometry_params):
        """Test concave profile creation."""
        x_gr, zb_gr, w_geom = create_concave_profile(**basic_geometry_params, perturbation=-200)
        
        # Concave profile should be lower in the middle than uniform
        _, zb_uniform, _ = create_uniform_slope(**basic_geometry_params)
        mid_point_idx = len(x_gr) // 2
        assert zb_gr[mid_point_idx] < zb_uniform[mid_point_idx]

    def test_create_convex_profile(self, basic_geometry_params):
        """Test convex profile creation."""
        x_gr, zb_gr, w_geom = create_convex_profile(**basic_geometry_params, perturbation=200)

        # Convex profile should be higher in the middle than uniform
        _, zb_uniform, _ = create_uniform_slope(**basic_geometry_params)
        mid_point_idx = len(x_gr) // 2
        assert zb_gr[mid_point_idx] > zb_uniform[mid_point_idx]

    def test_create_variable_width(self, basic_geometry_params):
        """Test variable width profile creation."""
        params = basic_geometry_params.copy()
        params.pop('width')
        x_gr, zb_gr, w_geom = create_variable_width(**params, w_head=2000, w_term=500)
        assert w_geom[0] == 2000
        assert w_geom[-1] == 500
        assert np.all(np.diff(w_geom) < 0)

    def test_create_spline_profile_basic(self, basic_geometry_params):
        """Test basic spline profile creation."""
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            z_start=2000, z_end=0, width=1000
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert zb_gr[0] == 2000  # Should match z_start exactly
        assert zb_gr[-1] == 0    # Should match z_end exactly
        assert np.all(w_geom == 1000)  # Constant width
        
    def test_create_spline_profile_with_control_points(self, basic_geometry_params):
        """Test spline profile with elevation control points."""
        control_points = [(5000, 1200), (10000, 400)]
        
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=100,  # More points for better interpolation accuracy
            z_start=2000, z_end=0,
            width=1000,
            control_points=control_points
        )
        
        # Check endpoints are nearly exact (allowing for floating point precision)
        assert abs(zb_gr[0] - 2000) < 1e-10
        assert abs(zb_gr[-1] - 0) < 1e-10
        
        # Check control points are approximately honored (within grid resolution)
        for x_pt, z_pt in control_points:
            idx = np.argmin(np.abs(x_gr - x_pt))
            z_actual = zb_gr[idx]
            assert abs(z_actual - z_pt) < 50  # Within 50m tolerance
    
    def test_create_spline_profile_variable_width_linear(self, basic_geometry_params):
        """Test spline profile with linear width variation."""
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            z_start=1500, z_end=200,
            w_start=2000, w_end=500
        )
        
        assert w_geom[0] == 2000
        assert w_geom[-1] == 500
        assert np.all(np.diff(w_geom) <= 0)  # Monotonically decreasing
        
    def test_create_spline_profile_width_control_points(self, basic_geometry_params):
        """Test spline profile with width control points."""
        width_points = [(0, 1800), (7500, 1200), (15000, 600)]
        
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=100,
            z_start=1800, z_end=100,
            width_control_points=width_points
        )
        
        # Check width control points
        for x_pt, w_pt in width_points:
            idx = np.argmin(np.abs(x_gr - x_pt))
            w_actual = w_geom[idx]
            assert abs(w_actual - w_pt) < 50  # Within 50m tolerance
            
        # Ensure all widths are positive
        assert np.all(w_geom > 0)
    
    def test_create_spline_profile_default_width(self, basic_geometry_params):
        """Test spline profile with default width."""
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            z_start=1500, z_end=0
        )
        
        assert np.all(w_geom == 1000)  # Default width
    
    def test_create_spline_profile_smoothing(self, basic_geometry_params):
        """Test spline profile with smoothing."""
        control_points = [(3000, 1400), (6000, 1000), (9000, 600), (12000, 300)]
        
        # Test both exact interpolation and smoothed approximation
        x_gr1, zb_gr1, _ = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=100, z_start=1800, z_end=0,
            control_points=control_points, smoothing=0, width=1000
        )
        
        x_gr2, zb_gr2, _ = create_spline_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=100, z_start=1800, z_end=0,
            control_points=control_points, smoothing=100.0, width=1000
        )
        
        # Smoothed version should be different from exact interpolation
        assert not np.allclose(zb_gr1, zb_gr2)
        
        # Exact interpolation should have precise endpoints
        assert abs(zb_gr1[0] - 1800) < 1e-10
        assert abs(zb_gr1[-1] - 0) < 1e-10
        
        # Smoothed version may deviate slightly from endpoints
        assert abs(zb_gr2[0] - 1800) < 50  # Allow larger deviation for smoothed
        assert abs(zb_gr2[-1] - 0) < 50
    
    def test_create_spline_profile_error_handling(self, basic_geometry_params):
        """Test error handling in spline profile creation."""
        
        # Test invalid elevation control points (outside domain)
        with pytest.raises(GeometryError, match="outside domain"):
            create_spline_profile(
                domain_extent=10000, x_gr_points=50,
                z_start=2000, z_end=0, width=1000,
                control_points=[(15000, 500)]  # Outside domain
            )
        
        # Test invalid width control points (outside domain)
        with pytest.raises(GeometryError, match="outside domain"):
            create_spline_profile(
                domain_extent=10000, x_gr_points=50,
                z_start=2000, z_end=0,
                width_control_points=[(15000, 1000)]
            )
        
        # Test negative width control points
        with pytest.raises(GeometryError, match="must be positive"):
            create_spline_profile(
                domain_extent=10000, x_gr_points=50,
                z_start=2000, z_end=0,
                width_control_points=[(5000, -500)]
            )
        
        # Test negative start/end widths
        with pytest.raises(GeometryError, match="must be positive"):
            create_spline_profile(
                domain_extent=10000, x_gr_points=50,
                z_start=2000, z_end=0,
                w_start=-1000, w_end=500
            )
        
        # Test duplicate x-coordinates in elevation control points
        with pytest.raises(GeometryError, match="duplicate x-coordinates"):
            create_spline_profile(
                domain_extent=10000, x_gr_points=50,
                z_start=2000, z_end=0, width=1000,
                control_points=[(5000, 1200), (5000, 1000)]
            )
    
    def test_create_function_profile_lambda(self, basic_geometry_params):
        """Test function profile creation with lambda function."""
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=lambda x: 2000 * np.exp(-x/5000),
            width=1000
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert zb_gr[0] == 2000  # Should start at 2000
        assert zb_gr[-1] < zb_gr[0]  # Should decrease
        assert np.all(w_geom == 1000)  # Constant width
        
    def test_create_function_profile_string_expression(self, basic_geometry_params):
        """Test function profile with string expression."""
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function="amplitude * sin(frequency * x / domain_extent) + base_height",
            width=1200,
            function_kwargs={'amplitude': 200, 'frequency': 2, 'base_height': 1000}
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert np.all(w_geom == 1200)  # Constant width
        # Check the function is evaluated correctly (sinusoidal pattern)
        expected = 200 * np.sin(2 * x_gr / basic_geometry_params['domain_extent']) + 1000
        np.testing.assert_allclose(zb_gr, expected, rtol=1e-10)
        
    def test_create_function_profile_width_function(self, basic_geometry_params):
        """Test function profile with width function."""
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=lambda x: 1500 * (1 - x/basic_geometry_params['domain_extent'])**2,
            width_function=lambda x: 1000 + 500 * (x/basic_geometry_params['domain_extent'])
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert w_geom[0] == 1000  # Start width
        assert w_geom[-1] == 1500  # End width
        assert np.all(np.diff(w_geom) >= 0)  # Monotonically increasing
        
    def test_create_function_profile_string_width_function(self, basic_geometry_params):
        """Test function profile with string width function.""" 
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=lambda x: 1800 - 0.1 * x,
            width_function="min_width + width_change * (x / domain_extent)**exponent",
            function_kwargs={'min_width': 800, 'width_change': 600, 'exponent': 0.5}
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert w_geom[0] == 800  # Start width
        expected_width = 800 + 600 * (x_gr / basic_geometry_params['domain_extent'])**0.5
        np.testing.assert_allclose(w_geom, expected_width, rtol=1e-10)
        
    def test_create_function_profile_linear_width(self, basic_geometry_params):
        """Test function profile with linear width variation."""
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=lambda x: 2000 - 0.12 * x,
            w_start=2000, w_end=500
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert w_geom[0] == 2000  # Start width
        assert w_geom[-1] == 500  # End width
        assert np.all(np.diff(w_geom) <= 0)  # Monotonically decreasing
        
    def test_create_function_profile_default_width(self, basic_geometry_params):
        """Test function profile with default width."""
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=lambda x: 1500 * np.exp(-x/8000)
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert np.all(w_geom == 1000.0)  # Default width
        
    def test_create_function_profile_step_function(self, basic_geometry_params):
        """Test function profile with step function."""
        step_func = lambda x: np.where(x < basic_geometry_params['domain_extent']/2, 1500, 800)
        
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=step_func,
            width=1200
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert np.all(w_geom == 1200)
        # Check step function behavior
        midpoint_idx = len(x_gr) // 2
        assert np.all(zb_gr[:midpoint_idx] == 1500)
        assert np.all(zb_gr[midpoint_idx:] == 800)
        
    def test_create_function_profile_error_handling(self, basic_geometry_params):
        """Test error handling in function profile creation."""
        
        # Test function that returns wrong shape
        def bad_shape_func(x):
            return np.array([1000])  # Wrong shape
            
        with pytest.raises(GeometryError, match="must return array with shape"):
            create_function_profile(
                domain_extent=basic_geometry_params['domain_extent'],
                x_gr_points=basic_geometry_params['x_gr_points'],
                elevation_function=bad_shape_func,
                width=1000
            )
            
        # Test function that returns NaN
        def nan_func(x):
            result = np.full_like(x, 1000.0)
            result[0] = np.nan
            return result
            
        with pytest.raises(GeometryError, match="non-finite values"):
            create_function_profile(
                domain_extent=basic_geometry_params['domain_extent'],
                x_gr_points=basic_geometry_params['x_gr_points'],
                elevation_function=nan_func,
                width=1000
            )
            
        # Test width function that returns negative values
        def negative_width_func(x):
            return -500 * np.ones_like(x)
            
        with pytest.raises(GeometryError, match="non-positive values"):
            create_function_profile(
                domain_extent=basic_geometry_params['domain_extent'],
                x_gr_points=basic_geometry_params['x_gr_points'],
                elevation_function=lambda x: 1000 * np.ones_like(x),
                width_function=negative_width_func
            )
            
        # Test invalid string expression
        with pytest.raises(GeometryError, match="evaluation failed"):
            create_function_profile(
                domain_extent=basic_geometry_params['domain_extent'],
                x_gr_points=basic_geometry_params['x_gr_points'],
                elevation_function="invalid_function(x)",
                width=1000
            )
            
        # Test negative start/end widths
        with pytest.raises(GeometryError, match="must be positive"):
            create_function_profile(
                domain_extent=basic_geometry_params['domain_extent'],
                x_gr_points=basic_geometry_params['x_gr_points'],
                elevation_function=lambda x: 1000 * np.ones_like(x),
                w_start=-1000, w_end=500
            )
    
    def test_create_function_profile_complex_mathematical_functions(self, basic_geometry_params):
        """Test function profile with complex mathematical expressions."""
        # Test combination of trigonometric and exponential functions
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=basic_geometry_params['domain_extent'],
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function="amplitude * exp(-x/decay_length) * cos(frequency * x/domain_extent) + baseline",
            width_function="max_width - width_decay * tanh(x/transition_length)",
            function_kwargs={
                'amplitude': 500,
                'decay_length': 8000,
                'frequency': 3,
                'baseline': 800,
                'max_width': 1800,
                'width_decay': 800,
                'transition_length': 4000
            }
        )
        
        assert len(x_gr) == basic_geometry_params['x_gr_points']
        assert np.all(np.isfinite(zb_gr))  # Should be finite
        assert np.all(w_geom > 0)  # Width should be positive
        assert w_geom[0] > w_geom[-1]  # Width should decrease
        
    def test_create_function_profile_comparison_with_existing_methods(self, basic_geometry_params):
        """Test that function profile can replicate existing geometry methods."""
        # Replicate uniform slope using function approach
        elevation_drop = basic_geometry_params['elevation_drop']
        domain_extent = basic_geometry_params['domain_extent']
        bed_length = basic_geometry_params['bed_characteristic_length']
        
        # Traditional method
        x_gr_trad, zb_gr_trad, w_geom_trad = create_uniform_slope(
            **basic_geometry_params
        )
        
        # Function method equivalent
        def uniform_slope_func(x):
            return elevation_drop * (1 - x / bed_length)
            
        x_gr_func, zb_gr_func, w_geom_func = create_function_profile(
            domain_extent=domain_extent,
            x_gr_points=basic_geometry_params['x_gr_points'],
            elevation_function=uniform_slope_func,
            width=basic_geometry_params['width']
        )
        
        # Should be nearly identical
        np.testing.assert_allclose(x_gr_trad, x_gr_func, rtol=1e-10)
        np.testing.assert_allclose(zb_gr_trad, zb_gr_func, rtol=1e-10)
        np.testing.assert_allclose(w_geom_trad, w_geom_func, rtol=1e-10)


class TestFlowlineGeometry:
    """Tests for the FlowlineGeometry class."""
    
    def test_geometry_interpolation(self, basic_geometry_params):
        """Test that geometry interpolates correctly to model grid."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=25)
        
        assert len(geometry.x) == len(geometry.zb)
        assert len(geometry.x) == len(geometry.w)
        assert np.all(np.diff(geometry.zb) <= 0)
        assert np.all(geometry.w > 0)

    def test_gradient_calculation(self, basic_geometry_params):
        """Test bed slope calculation."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)
        
        expected_slope = -basic_geometry_params['elevation_drop'] / basic_geometry_params['bed_characteristic_length']
        mean_slope = np.mean(geometry.dzbdx)
        assert abs(mean_slope - expected_slope) < 0.01

    def test_initial_profile_from_values(self, basic_geometry_params):
        """Test loading initial profile from h_init and x_init arrays."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        h_init = np.linspace(100, 0, len(x_gr))

        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
        geometry.setup_grid(delx=50)
        geometry.load_initial_profile()

        assert hasattr(geometry, 'h0')
        assert len(geometry.h0) == len(geometry.x)
        assert geometry.h0[0] > 0
    
    def test_no_initial_profile_raises_error(self, basic_geometry_params):
        """Test that not providing an initial profile raises an error."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)

        with pytest.raises(GeometryError, match="No valid initial profile"):
            geometry.load_initial_profile()
    
    def test_plot_geometry_basic(self, basic_geometry_params):
        """Test basic geometry plotting."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)
        
        # Test basic plot
        fig, axes = geometry.plot_geometry()
        assert len(axes) == 2  # bed elevation and width
        assert fig is not None
        
        # Close figure to avoid memory issues in tests
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_geometry_with_gradients(self, basic_geometry_params):
        """Test geometry plotting with gradients."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)
        
        # Test plot with gradients
        fig, axes = geometry.plot_geometry(show_gradients=True)
        assert len(axes) == 4  # bed, width, bed slope, width gradient
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_geometry_with_initial_profile(self, basic_geometry_params):
        """Test geometry plotting with initial profile."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        h_init = np.linspace(100, 0, len(x_gr))
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
        geometry.setup_grid(delx=50)
        geometry.load_initial_profile()
        
        # Test plot with initial profile
        fig, axes = geometry.plot_geometry(show_initial_profile=True)
        assert len(axes) == 3  # bed, width, initial profile
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_geometry_comprehensive(self, basic_geometry_params):
        """Test comprehensive geometry plotting."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        h_init = np.linspace(150, 0, len(x_gr))
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
        geometry.setup_grid(delx=50)
        geometry.load_initial_profile()
        
        # Test comprehensive plot
        fig, axes = geometry.plot_geometry(
            figsize=(14, 10),
            show_gradients=True, 
            show_initial_profile=True
        )
        assert len(axes) == 5  # bed, width, bed slope, width gradient, initial profile
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_geometry_error_handling(self, basic_geometry_params):
        """Test error handling in geometry plotting."""
        x_gr, zb_gr, w_geom = create_uniform_slope(**basic_geometry_params)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        
        # Test error when setup_grid hasn't been called
        with pytest.raises(GeometryError, match="Must call setup_grid"):
            geometry.plot_geometry()
    
    def test_plot_geometry_with_plan_view(self, basic_geometry_params):
        """Test geometry plotting with plan view."""
        params = basic_geometry_params.copy()
        params.pop('width')  # create_variable_width doesn't use this
        x_gr, zb_gr, w_geom = create_variable_width(**params, w_head=2000, w_term=500)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
        geometry.setup_grid(delx=50)
        
        # Test plot with plan view
        fig, axes = geometry.plot_geometry(show_plan_view=True, figsize=(14, 8))
        assert len(axes) == 3  # bed, width, plan view
        
        # Check that plan view axis has the expected properties
        plan_ax = axes[2]  # Plan view is now the third plot in single column
        assert 'Plan View' in plan_ax.get_title()
        assert 'Width extent [km]' in plan_ax.get_ylabel()
        # X-axis label is only on the bottom plot due to shared x-axis
        assert 'Distance from head [km]' in axes[-1].get_xlabel()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_geometry_comprehensive_with_plan_view(self, basic_geometry_params):
        """Test comprehensive geometry plotting including plan view."""
        params = basic_geometry_params.copy()
        params.pop('width')  # create_variable_width doesn't use this
        x_gr, zb_gr, w_geom = create_variable_width(**params, w_head=2000, w_term=500)
        h_init = np.linspace(200, 0, len(x_gr))
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_init=x_gr, h_init=h_init)
        geometry.setup_grid(delx=50)
        geometry.load_initial_profile()
        
        # Test comprehensive plot with plan view
        fig, axes = geometry.plot_geometry(
            figsize=(16, 12),
            show_gradients=True, 
            show_initial_profile=True,
            show_plan_view=True
        )
        # bed, width, bed slope, width gradient, initial profile, plan view
        assert len(axes) == 6
        
        # Plan view should be the last axis in single column layout
        plan_ax = axes[-1]
        assert 'Plan View' in plan_ax.get_title()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
