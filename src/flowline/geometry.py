import logging
import dill
import numpy as np
from scipy.interpolate import interp1d

# Custom exceptions
class FlowlineModelError(Exception):
    """Base exception for flowline model errors"""
    pass

class GeometryError(FlowlineModelError):
    """Errors related to geometry setup"""
    pass

class FlowlineGeometry:
    """Handles glacier geometry setup and interpolation"""
    
    def __init__(self, x_gr, zb_gr, w_geom, x_init=None, h_init=None, profile=None, profile_avg_nyears=None):
        self.x_gr = np.array(x_gr)
        self.zb_gr = np.array(zb_gr)
        self.w_geom = np.array(w_geom)
        self.x_init = x_init
        self.h_init = h_init
        self.profile = profile
        self.profile_avg_nyears = profile_avg_nyears
        
    def setup_grid(self, delx):
        """Create model grid and interpolate geometry"""
        xmx = np.max(delx * np.floor(self.x_gr / delx))
        self.x = np.arange(0, xmx, delx)
        self.nxs = len(self.x)
        
        # Interpolate bed elevation and width
        zb_interp = interp1d(self.x_gr, self.zb_gr)
        self.zb = zb_interp(self.x)
        
        w_interp = interp1d(self.x_gr, self.w_geom)
        self.w = w_interp(self.x)
        
        # Calculate gradients
        self.dzbdx = np.gradient(self.zb, self.x)
        self.dwdx = np.gradient(self.w, self.x)
        
        # Geometry validation
        if any(self.dzbdx == 0):
            logging.warning(f'Bed slope is zero at {(self.dzbdx == 0).argmax()}.')
        if any(self.dzbdx[0:2] > 0):
            logging.warning('The slope of the bed at the top of the glacier is positive. This may cause instabilities.')
    
    def load_initial_profile(self):
        """Load initial thickness profile, with optional averaging."""
        profile_source = None
        h0, x0 = None, None

        # 1. Try to load profile from a flowline2d object or a file
        if self.profile:
            if hasattr(self.profile, 'h') and hasattr(self.profile, 'x'):
                profile_source = self.profile
            else:
                try:
                    with open(self.profile, 'rb') as f:
                        profile_source = dill.load(f)
                    logging.info(f"Successfully loaded profile from: {self.profile}")
                except Exception:
                    profile_source = None

        # 2. Extract h0 from profile or use h_init
        if profile_source:
            x0 = np.array(profile_source.x)
            if self.profile_avg_nyears and self.profile_avg_nyears > 0 and len(profile_source.t) > 1:
                dt_out = np.mean(np.diff(profile_source.t))
                num_steps = int(round(self.profile_avg_nyears / dt_out))
                num_steps = max(1, num_steps)
                if num_steps > len(profile_source.t):
                    logging.warning(f"Cannot average over {self.profile_avg_nyears} years, only {len(profile_source.t)*dt_out:.1f} available. Averaging over entire profile history.")
                    num_steps = len(profile_source.t)
                h0 = np.mean(profile_source.h[-num_steps:, :], axis=0)
            else:
                h0 = np.array(profile_source.h[-1, :])
        elif self.x_init is not None and self.h_init is not None:  # In case x_init and h_init are explicitly provided
            logging.info("Using provided initial values for geometry.")
            x0 = self.x_init
            h0 = self.h_init
        else:
            raise GeometryError("No valid initial profile or initial values (x_init, h_init) provided.")

        # 3. Interpolate h0 to the model grid
        try:
            if np.any(self.x > x0.max()) or np.any(self.x < x0.min()):
                logging.warning(
                    f"Extrapolating h0 to model grid. x0 range: [{x0.min():.0f}, {x0.max():.0f}], x range: [{self.x.min():.0f}, {self.x.max():.0f}]"
                )
            h0_interp = interp1d(x0, h0, "linear", bounds_error=False, fill_value="extrapolate")
            self.h0 = h0_interp(self.x)
        except ValueError as e:
            raise GeometryError(f"Error during initial profile interpolation: {e}. x0 shape: {x0.shape}, h0 shape: {h0.shape}")

        return profile_source


def create_uniform_slope(domain_extent, x_gr_points, elevation_drop, width, bed_characteristic_length):
    """Create uniform slope bed profile"""
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    w_geom = np.full_like(x_gr, width)
    return x_gr, zb_gr, w_geom

def create_concave_profile(domain_extent, x_gr_points, elevation_drop, width, bed_characteristic_length, perturbation=-200):
    """Create slightly concave bed profile"""
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    # Base uniform slope
    zb_uniform = elevation_drop * (1 - x_gr / bed_characteristic_length)
    # Add concave perturbation
    perturb = perturbation * np.sin(np.pi * x_gr / bed_characteristic_length)**2
    zb_gr = zb_uniform + perturb
    w_geom = np.full_like(x_gr, width)
    return x_gr, zb_gr, w_geom

def create_convex_profile(domain_extent, x_gr_points, elevation_drop, width, bed_characteristic_length, perturbation=200):
    """Create slightly convex bed profile"""
    # Same as concave but with positive perturbation
    return create_concave_profile(domain_extent, x_gr_points, elevation_drop, width, bed_characteristic_length, perturbation)

def create_variable_width(domain_extent, x_gr_points, elevation_drop, bed_characteristic_length, w_head=2000, w_term=500):
    """Create variable width profile"""
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    # Width varies linearly
    w_geom = w_head - (w_head - w_term) * (x_gr / bed_characteristic_length)
    return x_gr, zb_gr, w_geom
