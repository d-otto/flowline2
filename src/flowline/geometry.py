"""
Glacier geometry setup and management module.

This module provides classes and functions for handling glacier bed geometry,
spatial grids, and initial ice thickness profiles. It supports both manual
geometry specification and predefined geometry functions for common research scenarios.

Key Components:
    - FlowlineGeometry: Main geometry management class
    - Geometry creation functions: Predefined bed profiles (uniform, concave, convex, variable width, spline-based, function-based)
    - Grid interpolation and validation utilities

Example Usage:
    # Create a uniform slope geometry
    x_gr, zb_gr, w_geom = create_uniform_slope(
        domain_extent=10000, x_gr_points=100,
        elevation_drop=2000, width=1000,
        bed_characteristic_length=8000
    )

    # Initialize geometry object with zero ice
    geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=np.zeros_like(x_gr))
    geometry.setup_grid(delx=25)
"""

import logging
import xarray as xr
from typing import Optional, Tuple, Union, Any, List, Callable
import numpy as np
from scipy.interpolate import interp1d, splrep, splev

# Custom exceptions
class FlowlineModelError(Exception):
    """Base exception for flowline model errors"""
    pass

class GeometryError(FlowlineModelError):
    """Errors related to geometry setup and interpolation"""
    pass

class FlowlineGeometry:
    """
    Manages glacier bed geometry, spatial grids, and initial ice thickness profiles.

    Attributes:
        x_gr (np.ndarray): High-resolution x-coordinates for geometry definition [m]
        zb_gr (np.ndarray): High-resolution bed elevation at x_gr [m]
        w_geom (np.ndarray): High-resolution channel width at x_gr [m]
        h0_gr (np.ndarray): Initial ice thickness on the x_gr grid [m]

    Grid Attributes (set after setup_grid):
        x (np.ndarray): Model x-coordinates [m]
        nxs (int): Number of grid points
        zb (np.ndarray): Bed elevation on model grid [m]
        w (np.ndarray): Channel width on model grid [m]
        dzbdx (np.ndarray): Bed slope (dz/dx) [dimensionless]
        dwdx (np.ndarray): Width gradient (dw/dx) [m/m]
        h0 (np.ndarray): Initial ice thickness on model grid [m]

    Example:
        # Create geometry with uniform slope and zero initial ice
        x_gr, zb_gr, w_geom = create_uniform_slope(10000, 100, 2000, 1000, 8000)
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, h0=np.zeros_like(x_gr))
        geometry.setup_grid(delx=25)
    """

    def __init__(self, x_gr: np.ndarray, zb_gr: np.ndarray, w_geom: np.ndarray,
                 h0: np.ndarray):
        """
        Initialize FlowlineGeometry with high-resolution geometry data.

        Args:
            x_gr: High-resolution x-coordinates for geometry definition [m]
            zb_gr: High-resolution bed elevation at x_gr [m]
            w_geom: High-resolution channel width at x_gr [m]
            h0: Initial ice thickness on the x_gr grid [m]. Pass np.zeros_like(x_gr)
                for a zero-ice start. Use FlowlineGeometry.from_profile() to load
                h0 from a previous run's NetCDF output.

        Note:
            Arrays x_gr, zb_gr, w_geom, and h0 must all have the same length and
            be sorted by increasing x_gr values for proper interpolation.
        """
        self.x_gr = np.array(x_gr)
        self.zb_gr = np.array(zb_gr)
        self.w_geom = np.array(w_geom)
        self.h0_gr = np.array(h0)
        
    def setup_grid(self, delx: float) -> None:
        """
        Create uniform model grid and interpolate geometry data onto it.

        Args:
            delx: Spatial grid spacing [m]. Typically 25-100m for flowline models.

        Sets:
            x: Model x-coordinates [m]
            nxs: Number of grid points
            zb: Bed elevation on model grid [m]
            w: Channel width on model grid [m]
            dzbdx: Bed slope (gradient) [dimensionless]
            dwdx: Width gradient [m/m]
            h0: Initial ice thickness on model grid [m]

        Note:
            Grid domain is automatically determined from geometry extent and delx.
            Issues warnings for zero bed slopes or positive slopes at glacier head.
        """
        xmx = np.max(delx * np.floor(self.x_gr / delx))
        self.x = np.arange(0, xmx, delx)
        self.nxs = len(self.x)

        # Interpolate bed elevation and width
        zb_interp = interp1d(self.x_gr, self.zb_gr)
        self.zb = zb_interp(self.x)

        w_interp = interp1d(self.x_gr, self.w_geom)
        self.w = w_interp(self.x)

        # Interpolate h0 onto model grid
        h0_interp = interp1d(self.x_gr, self.h0_gr, bounds_error=False, fill_value=0.0)
        self.h0 = np.maximum(0, h0_interp(self.x))

        # Calculate gradients
        self.dzbdx = np.gradient(self.zb, self.x)
        self.dwdx = np.gradient(self.w, self.x)

        # Geometry validation
        if any(self.dzbdx == 0):
            logging.warning(f'Bed slope is zero at {(self.dzbdx == 0).argmax()}.')
        if any(self.dzbdx[0:2] > 0):
            logging.warning('The slope of the bed at the top of the glacier is positive. This may cause instabilities.')

    @classmethod
    def from_profile(cls, path, x_gr, zb_gr, w_geom, avg_nyears=None):
        """
        Construct a FlowlineGeometry with h0 loaded from a previous run's NetCDF output.

        h is interpolated from the profile's grid directly onto x_gr during construction.
        The normal setup_grid interpolation (x_gr → model grid) then follows as usual.
        The profile grid does not need to match x_gr or delx.

        Parameters
        ----------
        path : str or Path
            Path to a NetCDF file produced by a previous flowline2d run.
        x_gr, zb_gr, w_geom : array
            High-resolution geometry arrays for the new run.
        avg_nyears : float, optional
            If given, average h over the final N years of the profile instead of
            using only the final timestep.
        """
        from pathlib import Path
        with xr.open_dataset(Path(path)) as ds:
            if avg_nyears is not None:
                dt = float(np.diff(ds['time'].values).mean())
                n = max(1, int(round(avg_nyears / dt)))
                h0_profile = ds['h'].isel(time=slice(-n, None)).mean(dim='time').values
            else:
                h0_profile = ds['h'].isel(time=-1).values
            x_profile = ds['x'].values

        h0_interp = interp1d(x_profile, h0_profile, bounds_error=False, fill_value=0.0)
        h0_on_xgr = np.maximum(0, h0_interp(np.asarray(x_gr)))
        return cls(x_gr, zb_gr, w_geom, h0=h0_on_xgr)
    
    def plot_geometry(self, figsize=(12, 8), show_gradients=False, show_initial_profile=False, 
                     show_plan_view=False):
        """
        Plot the glacier bed geometry and optionally width, gradients, initial profile, and plan view.
        
        Creates a comprehensive visualization of the geometry including bed elevation,
        channel width, spatial gradients, initial ice thickness, and top-down plan view.
        
        Args:
            figsize: Figure size tuple (width, height) in inches
            show_gradients: If True, plot bed slope and width gradient in additional subplots
            show_initial_profile: If True, plot initial ice thickness (requires h0 to be loaded)
            show_plan_view: If True, add plan view (top-down) showing glacier outline and bed elevation
        
        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects
        
        Raises:
            GeometryError: If setup_grid() hasn't been called yet
            
        Example:
            # Basic geometry plot
            geometry = FlowlineGeometry(x_gr, zb_gr, w_geom)
            geometry.setup_grid(delx=25)
            fig, axes = geometry.plot_geometry()
            
            # Comprehensive plot with all features including plan view
            fig, axes = geometry.plot_geometry(
                figsize=(16, 12), 
                show_gradients=True, 
                show_initial_profile=True,
                show_plan_view=True
            )
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise GeometryError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        # Check if grid has been set up
        if not hasattr(self, 'x') or not hasattr(self, 'zb'):
            raise GeometryError("Must call setup_grid() before plotting")
        
        # Determine subplot layout - 2x3 grid for cleaner layout
        n_plots = 2  # bed elevation and width
        if show_gradients:
            n_plots += 2  # add dzbdx and dwdx
        if show_initial_profile and hasattr(self, 'h0'):
            n_plots += 1  # add initial profile
        if show_plan_view:
            n_plots += 1  # add plan view
        
        # Use 2-column layout if we have more than 3 plots
        if n_plots <= 3:
            nrows, ncols = n_plots, 1
        else:
            nrows, ncols = 3, 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        
        # Handle axes indexing for different layouts
        if n_plots == 1:
            axes = [axes]
        elif ncols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
            
        # Hide unused subplots in 2x3 layout
        if ncols == 2 and n_plots < 6:
            for i in range(n_plots, 6):
                if i < len(axes):
                    axes[i].set_visible(False)
        
        plot_idx = 0
        
        # Plot bed elevation
        axes[plot_idx].plot(self.x / 1000, self.zb, 'k-', linewidth=2, label='Model grid')
        if hasattr(self, 'x_gr'):
            axes[plot_idx].plot(self.x_gr / 1000, self.zb_gr, 'k--', alpha=0.7, label='High-res')
        axes[plot_idx].set_ylabel('Bed elevation [m]')
        axes[plot_idx].set_title('Glacier bed geometry')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Plot channel width
        axes[plot_idx].plot(self.x / 1000, self.w, 'b-', linewidth=2, label='Model grid')
        if hasattr(self, 'x_gr'):
            axes[plot_idx].plot(self.x_gr / 1000, self.w_geom, 'b--', alpha=0.7, label='High-res')
        axes[plot_idx].set_ylabel('Channel width [m]')
        axes[plot_idx].set_title('Channel width')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Plot gradients if requested
        if show_gradients:
            # Bed slope
            axes[plot_idx].plot(self.x / 1000, self.dzbdx, 'r-', linewidth=2)
            axes[plot_idx].set_ylabel('Bed slope [dimensionless]')
            axes[plot_idx].set_title('Bed slope (dz/dx)')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axhline(y=0, color='k', linestyle=':', alpha=0.5)
            plot_idx += 1
            
            # Width gradient
            axes[plot_idx].plot(self.x / 1000, self.dwdx, 'g-', linewidth=2)
            axes[plot_idx].set_ylabel('Width gradient [m/m]')
            axes[plot_idx].set_title('Width gradient (dw/dx)')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axhline(y=0, color='k', linestyle=':', alpha=0.5)
            plot_idx += 1
        
        # Plot initial profile if requested and available
        if show_initial_profile and hasattr(self, 'h0'):
            # Plot both bed and surface elevation
            surface_elev = self.zb + self.h0
            axes[plot_idx].fill_between(self.x / 1000, self.zb, surface_elev, 
                                       alpha=0.6, color='lightblue', label='Ice thickness')
            axes[plot_idx].plot(self.x / 1000, self.zb, 'k-', linewidth=2, label='Bed')
            axes[plot_idx].plot(self.x / 1000, surface_elev, 'b-', linewidth=2, label='Surface')
            axes[plot_idx].set_ylabel('Elevation [m]')
            axes[plot_idx].set_title('Initial ice profile')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        elif show_initial_profile and not hasattr(self, 'h0'):
            # Add a note about missing initial profile
            axes[plot_idx].text(0.5, 0.5, 'Initial profile not available\n(h0 not set on model grid yet)',
                               transform=axes[plot_idx].transAxes, ha='center', va='center',
                               fontsize=12, style='italic', bbox=dict(boxstyle='round', facecolor='wheat'))
            axes[plot_idx].set_ylabel('Initial profile')
            axes[plot_idx].set_title('Initial ice profile (not available)')
            plot_idx += 1
        
        # Plot plan view if requested
        if show_plan_view:
            plan_ax = axes[plot_idx]
            plot_idx += 1
            
            # Create discrete elevation levels every 100m
            elevation_min = int(np.floor(self.zb.min() / 100) * 100)
            elevation_max = int(np.ceil(self.zb.max() / 100) * 100)
            clevels = np.arange(elevation_min, elevation_max + 100, 100)
            
            # Create coordinates for plan view
            x_2d = self.x / 1000  # Distance from head in km
            y_extent = self.w / 2000  # Half-width on each side in km
            
            # Create discretized colormap
            import matplotlib as mpl
            from matplotlib.colors import ListedColormap
            
            # Get base terrain colormap and discretize it  
            terrain_cmap = plt.cm.get_cmap('terrain')
            n_colors = len(clevels) - 1
            colors = [terrain_cmap(i / (n_colors - 1)) for i in range(n_colors)]
            discrete_cmap = ListedColormap(colors)
            norm = mpl.colors.BoundaryNorm(clevels, discrete_cmap.N)
            
            # Create filled regions showing glacier outline with discrete elevation colors
            for i in range(len(x_2d) - 1):
                # Get average bed elevation for this segment
                bed_elev_avg = (self.zb[i] + self.zb[i+1]) / 2
                
                # Find which color bin this elevation falls into
                color_idx = np.digitize(bed_elev_avg, clevels) - 1
                color_idx = np.clip(color_idx, 0, len(colors) - 1)
                color = colors[color_idx]
                
                # Create rectangular patch for this segment
                x_vals = [x_2d[i], x_2d[i+1], x_2d[i+1], x_2d[i]]
                y_vals = [-y_extent[i], -y_extent[i+1], y_extent[i+1], y_extent[i]]
                
                # Fill the glacier segment
                plan_ax.fill(x_vals, y_vals, color=color, alpha=0.8, 
                           edgecolor='k', linewidth=0.1)
            
            # Create a dummy mappable for colorbar
            sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
            sm.set_array([])
            
            # Draw glacier outline
            plan_ax.plot(x_2d, y_extent, 'k-', linewidth=2, alpha=0.8)
            plan_ax.plot(x_2d, -y_extent, 'k-', linewidth=2, alpha=0.8)
            
            # Add ice extent if available
            if hasattr(self, 'h0'):
                ice_mask = self.h0 > 1  # Minimum 1m thickness
                if np.any(ice_mask):
                    ice_indices = np.where(ice_mask)[0]
                    if len(ice_indices) > 0:
                        ice_start_idx = ice_indices[0]
                        ice_end_idx = ice_indices[-1]
                        
                        # Draw ice extent markers
                        x_ice = x_2d[ice_start_idx:ice_end_idx+1]
                        y_ice_pos = y_extent[ice_start_idx:ice_end_idx+1]
                        y_ice_neg = -y_extent[ice_start_idx:ice_end_idx+1]
                        
                        plan_ax.plot(x_ice, y_ice_pos, 'b-', linewidth=3, 
                               label='Ice margin', alpha=0.8)
                        plan_ax.plot(x_ice, y_ice_neg, 'b-', linewidth=3, alpha=0.8)
            
            # Formatting for plan view
            plan_ax.set_ylabel('Width extent [km]')
            plan_ax.set_title('Plan View (Top-Down)')
            plan_ax.grid(True, alpha=0.3)
            plan_ax.set_aspect('equal')
            
            # Add discrete colorbar
            cbar = plt.colorbar(sm, ax=plan_ax, shrink=0.8)
            cbar.set_label('Bed elevation [m]')
            cbar.set_ticks(clevels[::2])  # Show every other level to avoid crowding
            
            if hasattr(self, 'h0') and np.any(self.h0 > 1):
                plan_ax.legend(loc='upper right')
        
        # Set x-axis labels for bottom plots (shared x-axis)
        if ncols == 1:
            axes[-1].set_xlabel('Distance from head [km]')
        else:
            # For 2-column layout, set x-labels on bottom row plots
            for i in range(ncols):
                bottom_idx = (nrows - 1) * ncols + i
                if bottom_idx < n_plots:
                    axes[bottom_idx].set_xlabel('Distance from head [km]')
        
        # Adjust layout with padding for titles
        plt.tight_layout(pad=2.0)
        
        return fig, axes


def create_uniform_slope(domain_extent: float, x_gr_points: int, elevation_drop: float, 
                        width: float, bed_characteristic_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform slope bed profile with constant width.
    
    This function generates a simple linear bed profile commonly used for 
    idealized glacier modeling experiments and sensitivity studies.
    
    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        width: Constant channel width [m]
        bed_characteristic_length: Distance over which elevation drops [m]
    
    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization
            x_gr: X-coordinates [m]
            zb_gr: Bed elevation [m] 
            w_geom: Channel width [m]
    
    Example:
        # Create 10 km domain with 2000 m elevation drop over 8 km
        x_gr, zb_gr, w_geom = create_uniform_slope(
            domain_extent=10000, x_gr_points=100,
            elevation_drop=2000, width=1000, 
            bed_characteristic_length=8000
        )
    """
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    w_geom = np.full_like(x_gr, width)
    return x_gr, zb_gr, w_geom

def create_concave_profile(domain_extent: float, x_gr_points: int, elevation_drop: float, 
                          width: float, bed_characteristic_length: float, perturbation: float = -200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a concave (overdeepened) bed profile with constant width.
    
    This function generates a bed profile with a uniform slope base plus a 
    sinusoidal perturbation that creates an overdeepened (concave) shape.
    Common for studying glacier dynamics in overdeepened valleys.
    
    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        width: Constant channel width [m]
        bed_characteristic_length: Distance over which elevation drops [m]
        perturbation: Amplitude of concave perturbation [m]. Negative creates overdeepening.
    
    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization
            x_gr: X-coordinates [m]
            zb_gr: Bed elevation with concave perturbation [m] 
            w_geom: Channel width [m]
    
    Example:
        # Create overdeepened profile with 200m depression
        x_gr, zb_gr, w_geom = create_concave_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_drop=2000, width=1000,
            bed_characteristic_length=8000, perturbation=-200
        )
    """
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    # Base uniform slope
    zb_uniform = elevation_drop * (1 - x_gr / bed_characteristic_length)
    # Add concave perturbation
    perturb = perturbation * np.sin(np.pi * x_gr / bed_characteristic_length)**2
    zb_gr = zb_uniform + perturb
    w_geom = np.full_like(x_gr, width)
    return x_gr, zb_gr, w_geom

def create_convex_profile(domain_extent: float, x_gr_points: int, elevation_drop: float, 
                         width: float, bed_characteristic_length: float, perturbation: float = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a convex (elevated) bed profile with constant width.
    
    This function generates a bed profile with a uniform slope base plus a 
    sinusoidal perturbation that creates an elevated (convex) shape.
    Useful for studying glacier dynamics over bedrock bumps or ridges.
    
    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        width: Constant channel width [m]
        bed_characteristic_length: Distance over which elevation drops [m]
        perturbation: Amplitude of convex perturbation [m]. Positive creates elevation.
    
    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization
            x_gr: X-coordinates [m]
            zb_gr: Bed elevation with convex perturbation [m] 
            w_geom: Channel width [m]
    
    Example:
        # Create elevated profile with 200m bump
        x_gr, zb_gr, w_geom = create_convex_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_drop=2000, width=1000,
            bed_characteristic_length=8000, perturbation=200
        )
    """
    # Same as concave but with positive perturbation
    return create_concave_profile(domain_extent, x_gr_points, elevation_drop, width, bed_characteristic_length, perturbation)

def create_variable_width(domain_extent: float, x_gr_points: int, elevation_drop: float,
                         bed_characteristic_length: float, w_head: float = 2000, w_term: float = 500,
                         w_mid: Optional[float] = None, x_mid: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform slope bed profile with linearly varying width.

    Width varies linearly from head to terminus. Optionally, a midpoint breakpoint
    (w_mid, x_mid) can be specified to create a piecewise linear width profile with
    two segments: head→mid and mid→terminus.

    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        bed_characteristic_length: Distance over which elevation drops [m]
        w_head: Channel width at glacier head (x=0) [m]
        w_term: Channel width at terminus (x=domain_extent) [m] when w_mid/x_mid are
            used; otherwise at x=bed_characteristic_length for the single-segment case.
        w_mid: Channel width at the breakpoint x_mid [m]. Requires x_mid.
        x_mid: X-coordinate of the width breakpoint [m]. Requires w_mid.

    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization

    Examples:
        # Narrowing channel
        x_gr, zb_gr, w_geom = create_variable_width(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=10000,
            w_head=2000, w_term=500
        )

        # Hourglass: wide → narrow at 4km → wide again
        x_gr, zb_gr, w_geom = create_variable_width(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=10000,
            w_head=2000, w_term=2000, w_mid=500, x_mid=4000
        )
    """
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    if w_mid is not None and x_mid is not None:
        w_geom = np.where(
            x_gr <= x_mid,
            w_head + (w_mid - w_head) * (x_gr / x_mid),
            w_mid  + (w_term - w_mid) * ((x_gr - x_mid) / (domain_extent - x_mid)),
        )
    else:
        w_geom = w_head - (w_head - w_term) * (x_gr / bed_characteristic_length)
    return x_gr, zb_gr, w_geom


def create_harmonic_bed(
    domain_extent: float,
    x_gr_points: int,
    elevation_drop: float,
    bed_characteristic_length: float,
    bed_perturbation: float,
    width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform slope bed with a sin^2 perturbation and constant width.

    Bed elevation is:
        zb(x) = zb_uniform(x) + bed_perturbation * sin(pi * x / bed_characteristic_length)^2

    The perturbation shape sin^2(pi*x/L) is zero at head (x=0) and terminus (x=L),
    and peaks at the center (x=L/2). Its integral over [0, L] equals L/2, so:
        bed_perturbation = 2 * target_integral / L

    Positive bed_perturbation produces a convex bed (bump at center).
    Negative bed_perturbation produces a concave bed (overdeepening at center).

    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        bed_characteristic_length: Period of the sin^2 shape and the linear slope length [m]
        bed_perturbation: Amplitude of bed perturbation [m].
            Positive = convex (elevated center), negative = concave (overdeepened center).
        width: Constant channel width [m]

    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization

    Examples:
        # Convex bed (bump at center)
        x_gr, zb_gr, w_geom = create_harmonic_bed(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=8000,
            bed_perturbation=50, width=1250
        )

        # Concave bed (overdeepening at center)
        x_gr, zb_gr, w_geom = create_harmonic_bed(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=8000,
            bed_perturbation=-50, width=1250
        )
    """
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    L = bed_characteristic_length
    zb_gr = zb_gr + bed_perturbation * np.sin(np.pi * x_gr / L) ** 2
    w_geom = np.full_like(x_gr, float(width))
    return x_gr, zb_gr, w_geom


def create_harmonic_width(
    domain_extent: float,
    x_gr_points: int,
    elevation_drop: float,
    bed_characteristic_length: float,
    harmonics: List[Tuple[int, float, float]],
    offset: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform slope bed with a cosine harmonic width profile.

    Width is defined as:
        w(x) = offset + sum_i( R_i * cos(n_i * 2*pi*x / bed_characteristic_length + phi_i) )

    One full cosine period spans bed_characteristic_length. Because each harmonic
    integrates to zero over a full period, the mean width equals `offset` regardless
    of the harmonic amplitudes, making it easy to compare shapes with equal cross-
    sectional area integrals.

    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_drop: Total elevation change from head to terminus [m]
        bed_characteristic_length: Distance over which elevation drops, and the
            period of the cosine harmonics [m]
        harmonics: List of (n, R, phi) tuples.
            n   — harmonic number (1=fundamental, 2=second harmonic, ...)
            R   — amplitude [m]
            phi — phase [radians]
        offset: Constant term added to all width values [m]. With offset=0 and a
            single harmonic, the width oscillates between -R and +R (centered at 0).
            Must be large enough that w(x) > 0 everywhere.

    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization

    Raises:
        ValueError: If any width value is <= 0.

    Examples:
        # Hourglass: wide at head/terminus (x=0, x=L), narrow at center
        x_gr, zb_gr, w_geom = create_harmonic_width(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=8000,
            harmonics=[(1, 750, 0)], offset=1250
        )

        # Oval: narrow at head/terminus, wide at center
        x_gr, zb_gr, w_geom = create_harmonic_width(
            domain_extent=12000, x_gr_points=61,
            elevation_drop=1000, bed_characteristic_length=8000,
            harmonics=[(1, 750, np.pi)], offset=1250
        )
    """
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    zb_gr = elevation_drop * (1 - x_gr / bed_characteristic_length)
    w_geom = np.full_like(x_gr, float(offset))
    for n, R, phi in harmonics:
        w_geom = w_geom + R * np.cos(n * 2 * np.pi * x_gr / bed_characteristic_length + phi)
    if np.any(w_geom <= 0):
        raise ValueError(
            f"Width profile has non-positive values (min={w_geom.min():.1f} m). "
            "Increase offset or reduce amplitude."
        )
    return x_gr, zb_gr, w_geom


def create_spline_profile(domain_extent: float, x_gr_points: int,
                         z_start: float, z_end: float,
                         width: Optional[float] = None,
                         width_control_points: Optional[List[Tuple[float, float]]] = None,
                         w_start: Optional[float] = None,
                         w_end: Optional[float] = None,
                         control_points: Optional[List[Tuple[float, float]]] = None,
                         smoothing: float = 0, spline_degree: int = 3,
                         **spline_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a bed profile using spline interpolation with flexible width handling.
    
    This function generates smooth bed and width profiles using scipy's spline interface,
    allowing precise control over geometry through required pass-through points while
    maintaining smoothness. Supports multiple width specification methods from simple
    constant width to complex spline-based width variations.
    
    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        z_start: Bed elevation at glacier head (x=0) [m]
        z_end: Bed elevation at domain end [m]
        width: Constant channel width [m]. If specified, overrides other width options.
        width_control_points: List of (x, width) tuples for spline-based width [m]
        w_start: Channel width at glacier head for linear variation [m]
        w_end: Channel width at terminus for linear variation [m]
        control_points: List of (x, z) tuples that elevation spline must pass through [m]
        smoothing: Smoothing factor for splines (0=interpolation, >0=approximation)
        spline_degree: Degree of spline (1=linear, 2=quadratic, 3=cubic)
        **spline_kwargs: Additional parameters passed to scipy.splrep
    
    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization
            x_gr: X-coordinates [m]
            zb_gr: Bed elevation from spline interpolation [m]
            w_geom: Channel width [m]
    
    Width Priority (first valid option used):
        1. width: Constant width across domain
        2. width_control_points: Spline interpolation through width control points
        3. w_start + w_end: Linear interpolation between start/end widths
        4. Default: 1000m constant width
    
    Raises:
        GeometryError: If control points are invalid or spline fitting fails
    
    Examples:
        # Simple smooth profile with constant width
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=10000, x_gr_points=100,
            z_start=2000, z_end=0, width=1000
        )
        
        # Complex bed with overdeepening and variable width
        bed_points = [(2000, 1500), (4000, 800), (6000, 200)]
        width_points = [(0, 2000), (5000, 1200), (10000, 500)]
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=10000, x_gr_points=100,
            z_start=2000, z_end=0,
            control_points=bed_points,
            width_control_points=width_points
        )
        
        # Linear width variation (like create_variable_width)
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=10000, x_gr_points=100,
            z_start=2000, z_end=0,
            w_start=2000, w_end=500
        )
        
        # Advanced: weighted control points and boundary conditions
        x_gr, zb_gr, w_geom = create_spline_profile(
            domain_extent=10000, x_gr_points=100,
            z_start=2000, z_end=0,
            control_points=[(3000, 1200), (7000, 400)],
            width=1000,
            smoothing=10.0,  # Allow small deviations for smoothness
            w=[1, 3, 2, 1]   # Weights passed to splrep
        )
    """
    # Create output x coordinates
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    
    # === BED ELEVATION SPLINE ===
    # Build list of all elevation control points including start/end
    elevation_points = [(0, z_start), (domain_extent, z_end)]
    
    if control_points:
        # Validate control points are within domain
        for x_pt, z_pt in control_points:
            if x_pt < 0 or x_pt > domain_extent:
                raise GeometryError(f"Elevation control point x={x_pt} is outside domain [0, {domain_extent}]")
        
        # Add control points and sort by x-coordinate
        elevation_points.extend(control_points)
        elevation_points.sort(key=lambda pt: pt[0])
    
    # Extract x and z coordinates for spline fitting
    x_ctrl_elev = np.array([pt[0] for pt in elevation_points])
    z_ctrl = np.array([pt[1] for pt in elevation_points])
    
    # Check for duplicate x values
    if len(np.unique(x_ctrl_elev)) != len(x_ctrl_elev):
        raise GeometryError("Elevation control points contain duplicate x-coordinates")
    
    try:
        # Fit elevation spline
        if smoothing == 0:
            # Exact interpolation through all points
            elev_tck = splrep(x_ctrl_elev, z_ctrl, 
                             k=min(spline_degree, len(x_ctrl_elev)-1), 
                             s=0, **spline_kwargs)
        else:
            # Smoothed approximation
            elev_tck = splrep(x_ctrl_elev, z_ctrl, 
                             k=min(spline_degree, len(x_ctrl_elev)-1), 
                             s=smoothing, **spline_kwargs)
        
        # Evaluate elevation spline at output points
        zb_gr = splev(x_gr, elev_tck)
        
    except Exception as e:
        raise GeometryError(f"Elevation spline fitting failed: {e}")
    
    # === WIDTH HANDLING ===
    if width is not None:
        # Option 1: Constant width
        w_geom = np.full_like(x_gr, width)
        
    elif width_control_points is not None:
        # Option 2: Spline-based width
        # Validate width control points
        for x_pt, w_pt in width_control_points:
            if x_pt < 0 or x_pt > domain_extent:
                raise GeometryError(f"Width control point x={x_pt} is outside domain [0, {domain_extent}]")
            if w_pt <= 0:
                raise GeometryError(f"Width control point w={w_pt} must be positive")
        
        # Sort by x-coordinate
        width_points_sorted = sorted(width_control_points, key=lambda pt: pt[0])
        x_ctrl_width = np.array([pt[0] for pt in width_points_sorted])
        w_ctrl = np.array([pt[1] for pt in width_points_sorted])
        
        # Check for duplicate x values
        if len(np.unique(x_ctrl_width)) != len(x_ctrl_width):
            raise GeometryError("Width control points contain duplicate x-coordinates")
        
        try:
            # Fit width spline (use same parameters as elevation spline)
            if smoothing == 0:
                width_tck = splrep(x_ctrl_width, w_ctrl, 
                                  k=min(spline_degree, len(x_ctrl_width)-1), 
                                  s=0, **spline_kwargs)
            else:
                width_tck = splrep(x_ctrl_width, w_ctrl, 
                                  k=min(spline_degree, len(x_ctrl_width)-1), 
                                  s=smoothing, **spline_kwargs)
            
            # Evaluate width spline
            w_geom = splev(x_gr, width_tck)
            
            # Ensure positive widths
            if np.any(w_geom <= 0):
                raise GeometryError("Spline-interpolated width becomes negative or zero")
                
        except Exception as e:
            raise GeometryError(f"Width spline fitting failed: {e}")
    
    elif w_start is not None and w_end is not None:
        # Option 3: Linear width variation (like create_variable_width)
        if w_start <= 0 or w_end <= 0:
            raise GeometryError("Start and end widths must be positive")
        w_geom = w_start + (w_end - w_start) * (x_gr / domain_extent)
        
    else:
        # Option 4: Default constant width
        w_geom = np.full_like(x_gr, 1000.0)
    
    return x_gr, zb_gr, w_geom


def create_function_profile(domain_extent: float, x_gr_points: int,
                          elevation_function: Union[Callable[[np.ndarray], np.ndarray], str],
                          width: Optional[float] = None,
                          width_function: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None,
                          w_start: Optional[float] = None,
                          w_end: Optional[float] = None,
                          function_kwargs: Optional[dict] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create geometry profiles from user-defined mathematical functions.
    
    This function enables flexible bed elevation and width specification through 
    Python callables, lambda functions, or string expressions. It provides a 
    powerful interface for creating complex mathematical bed profiles while 
    maintaining compatibility with the existing geometry creation workflow.
    
    Args:
        domain_extent: Total model domain length [m]
        x_gr_points: Number of high-resolution grid points for geometry definition
        elevation_function: Function or string expression for bed elevation z(x) [m]
        width: Constant channel width [m]. If specified, overrides other width options.
        width_function: Function or string expression for channel width w(x) [m]
        w_start: Channel width at glacier head for linear variation [m]
        w_end: Channel width at terminus for linear variation [m]
        function_kwargs: Dictionary of parameters passed to string expressions
    
    Returns:
        tuple: (x_gr, zb_gr, w_geom) arrays for FlowlineGeometry initialization
            x_gr: X-coordinates [m]
            zb_gr: Bed elevation from function evaluation [m]
            w_geom: Channel width [m]
    
    Width Priority (first valid option used):
        1. width: Constant width across domain
        2. width_function: Function-based width variation
        3. w_start + w_end: Linear interpolation between start/end widths
        4. Default: 1000m constant width
    
    String Expression Context:
        String expressions are evaluated with numpy context plus:
        - x: X-coordinate array [m]
        - domain_extent: Total domain length [m]
        - Any parameters from function_kwargs
        
        Mathematical functions available: all numpy functions (sin, cos, exp, log, etc.)
    
    Raises:
        GeometryError: If function evaluation fails or produces invalid values
        GeometryError: If width functions produce non-positive values
    
    Examples:
        # Exponential decay bed profile
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_function=lambda x: 2000 * np.exp(-x/5000),
            width=1000
        )
        
        # Parametric sinusoidal bed with function-based width
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_function="amplitude * np.sin(frequency * x / domain_extent) + base_slope * (domain_extent - x)",
            width_function=lambda x: 1500 - 500 * (x / domain_extent)**0.5,
            function_kwargs={'amplitude': 200, 'frequency': 2, 'base_slope': 0.15}
        )
        
        # Parabolic bed with linear width transition
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_function="start_elev * (1 - (x / domain_extent)**exponent)",
            w_start=2000, w_end=500,
            function_kwargs={'start_elev': 1800, 'exponent': 2.0}
        )
        
        # Step function bed profile
        step_func = lambda x: np.where(x < 5000, 1500, 800)
        x_gr, zb_gr, w_geom = create_function_profile(
            domain_extent=10000, x_gr_points=100,
            elevation_function=step_func,
            width=1200
        )
    """
    # Create output x coordinates
    x_gr = np.linspace(0, domain_extent, int(x_gr_points))
    
    # Initialize function_kwargs if not provided
    if function_kwargs is None:
        function_kwargs = {}
    
    # === BED ELEVATION FROM FUNCTION ===
    try:
        if isinstance(elevation_function, str):
            # String expression evaluation
            eval_context = {
                'x': x_gr,
                'domain_extent': domain_extent,
                'np': np,
                **function_kwargs
            }
            
            # Create safe evaluation context with numpy functions
            safe_context = {
                '__builtins__': {},
                **{name: getattr(np, name) for name in dir(np) if not name.startswith('_')},
                **eval_context
            }
            
            zb_gr = eval(elevation_function, safe_context)
            zb_gr = np.array(zb_gr)  # Ensure numpy array
            
        else:
            # Callable function (lambda or regular function)
            zb_gr = elevation_function(x_gr)
            zb_gr = np.array(zb_gr)  # Ensure numpy array
            
    except Exception as e:
        raise GeometryError(f"Elevation function evaluation failed: {e}")
    
    # Validate elevation output
    if zb_gr.shape != x_gr.shape:
        raise GeometryError(f"Elevation function must return array with shape {x_gr.shape}, got {zb_gr.shape}")
    
    if not np.isfinite(zb_gr).all():
        raise GeometryError("Elevation function produced non-finite values (NaN or Inf)")
    
    # === WIDTH HANDLING ===
    if width is not None:
        # Option 1: Constant width
        w_geom = np.full_like(x_gr, width)
        
    elif width_function is not None:
        # Option 2: Function-based width
        try:
            if isinstance(width_function, str):
                # String expression evaluation
                eval_context = {
                    'x': x_gr,
                    'domain_extent': domain_extent,
                    'np': np,
                    **function_kwargs
                }
                
                safe_context = {
                    '__builtins__': {},
                    **{name: getattr(np, name) for name in dir(np) if not name.startswith('_')},
                    **eval_context
                }
                
                w_geom = eval(width_function, safe_context)
                w_geom = np.array(w_geom)  # Ensure numpy array
                
            else:
                # Callable function
                w_geom = width_function(x_gr)
                w_geom = np.array(w_geom)  # Ensure numpy array
                
        except Exception as e:
            raise GeometryError(f"Width function evaluation failed: {e}")
            
        # Validate width output
        if w_geom.shape != x_gr.shape:
            raise GeometryError(f"Width function must return array with shape {x_gr.shape}, got {w_geom.shape}")
        
        if not np.isfinite(w_geom).all():
            raise GeometryError("Width function produced non-finite values (NaN or Inf)")
            
        if np.any(w_geom <= 0):
            raise GeometryError("Width function produced non-positive values")
            
    elif w_start is not None and w_end is not None:
        # Option 3: Linear width variation
        if w_start <= 0 or w_end <= 0:
            raise GeometryError("Start and end widths must be positive")
        w_geom = w_start + (w_end - w_start) * (x_gr / domain_extent)
        
    else:
        # Option 4: Default constant width
        w_geom = np.full_like(x_gr, 1000.0)
    
    return x_gr, zb_gr, w_geom
