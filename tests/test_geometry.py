import pytest
import numpy as np
import tempfile
import os
import dill

# Add src to path to find flowline modules
import sys
sys.path.append('src')

from flowline.geometry import (
    FlowlineGeometry, GeometryError, create_uniform_slope,
    create_concave_profile, create_convex_profile, create_variable_width
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
