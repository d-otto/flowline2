import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.colors as mcolors

import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from skimage import graph
from shapely.geometry import Polygon, Point, LineString
import warnings


class GlacierCenterlineExtractor:
    """
    Implementation of the glacier centerline extraction algorithm from:
    Kienholz et al. (2014) - A new method for deriving glacier centerlines
    applied to glaciers in Alaska and northwest Canada
    """

    def __init__(self, glacier_outline, dem, resolution=10):
        """
        Initialize the centerline extractor.

        Parameters:
        -----------
        glacier_outline : array-like
            N x 2 array of (x, y) coordinates defining the glacier outline
        dem : 2D numpy array
            Digital elevation model covering the glacier area
        resolution : float
            Grid cell size in meters (default: 10m)
        """
        self.outline = np.array(glacier_outline)
        self.dem = dem
        self.resolution = resolution

        # Create glacier polygon
        self.glacier_poly = Polygon(self.outline)

        # Algorithm parameters from Table 1 in the paper
        self.params = {
            "q1": 2e-6,  # For minimum head distance
            "q2": 500,  # For minimum head distance
            "rmax": 1000,  # Maximum head separation
            "f1": 1000,  # Euclidean distance weight
            "f2": 3000,  # Elevation weight
            "a": 4.25,  # Euclidean distance exponent
            "b": 3.5,  # Elevation exponent (initial)
            "u1": 2e-6,  # For smoothing length
            "u2": 200,  # For smoothing length
            "lmax": 400,  # Maximum smoothing length
            "j1": 0.1,  # For optimization threshold
            "j2": 0.01,  # For optimization threshold
            "j3": 0.7,  # For optimization threshold
            "w1": 1e-6,  # For branch buffer width
            "w2": 150,  # For branch buffer width
            "kmax": 650,  # Maximum branch buffer
        }

    def extract_centerlines(self):
        """
        Main method to extract glacier centerlines following the 3-step algorithm.

        Returns:
        --------
        centerlines : list of arrays
            List of centerline coordinates (each as N x 2 array)
        branches : list of dict
            List of branch information including order
        """
        # Step 1: Identify glacier heads and terminus
        terminus = self._find_terminus()
        heads = self._find_heads()

        # Step 2: Create cost grid and derive centerlines
        # Convert terminus from grid indices to real world coordinates
        # We need to use the actual coordinate bounds of the glacier outline
        outline_coords = self.outline
        coords_2d = outline_coords[:, :2] if outline_coords.shape[1] > 2 else outline_coords
        x_min, y_min = coords_2d.min(axis=0)
        x_max, y_max = coords_2d.max(axis=0)
        
        # Convert terminus grid indices to real coordinates
        dem_height, dem_width = self.dem.shape
        terminus_x = x_min + terminus[1] * (x_max - x_min) / dem_width
        terminus_y = y_max - terminus[0] * (y_max - y_min) / dem_height
        terminus_coord = (terminus_x, terminus_y)
        
        centerlines = []
        for head in heads:
            cost_grid = self._create_cost_grid()
            centerline = self._compute_least_cost_route(head, terminus_coord, cost_grid)

            # Optimization step
            optimized_cl = self._optimize_centerline(centerline, cost_grid)
            centerlines.append(optimized_cl)

        # Step 3: Derive branches and branch order
        branches = self._derive_branches(centerlines)

        return centerlines, branches

    def _find_terminus(self):
        """
        Find the glacier terminus as the lowest elevation point.

        Returns:
        --------
        terminus : tuple
            (x, y) coordinates of the terminus
        """
        # Create mask for glacier area
        mask = self._create_glacier_mask()

        # Apply low-pass filter to DEM
        filtered_dem = ndimage.gaussian_filter(self.dem, sigma=2)

        # Fill depressions
        filled_dem = self._fill_depressions(filtered_dem)

        # Find lowest point within glacier
        glacier_dem = np.where(mask, filled_dem, np.inf)
        min_idx = np.unravel_index(np.argmin(glacier_dem), glacier_dem.shape)

        return min_idx

    def _find_heads(self):
        """
        Find glacier heads as local elevation maxima along the outline.

        Returns:
        --------
        heads : list of tuples
            List of (x, y) coordinates for glacier heads
        """
        heads = []

        # Sample outline at 100m intervals
        outline_length = self._compute_outline_length()
        sample_dist = 100  # meters
        n_samples = max(int(outline_length / sample_dist), 3)  # Ensure minimum samples

        # Sample elevations along outline
        sampled_points = []
        sampled_elevs = []

        for i in range(n_samples):
            idx = int(i * len(self.outline) / n_samples)
            point = self.outline[idx]

            # Get elevation at this point
            grid_x = int(point[0] / self.resolution)
            grid_y = int(point[1] / self.resolution)

            if 0 <= grid_x < self.dem.shape[1] and 0 <= grid_y < self.dem.shape[0]:
                elev = self.dem[grid_y, grid_x]
                sampled_points.append(point)
                sampled_elevs.append(elev)

        # Check if we have any valid samples
        if len(sampled_elevs) == 0:
            # No valid elevation samples - use outline endpoints as heads
            return [self.outline[0], self.outline[len(self.outline)//2]]

        sampled_elevs = np.array(sampled_elevs)

        # Find local maxima (higher than 5 neighbors on each side)
        window = min(5, len(sampled_elevs) // 3)  # Adjust window size for small samples
        if window < 1:
            window = 1

        for i in range(window, len(sampled_elevs) - window):
            left_neighbors = sampled_elevs[max(0, i - window) : i]
            right_neighbors = sampled_elevs[i + 1 : min(len(sampled_elevs), i + window + 1)]

            if (len(left_neighbors) > 0 and len(right_neighbors) > 0 and
                sampled_elevs[i] > np.max(left_neighbors) and 
                sampled_elevs[i] > np.max(right_neighbors)):
                # Check if above lower third of elevation range
                elev_threshold = np.percentile(sampled_elevs, 33)
                if sampled_elevs[i] > elev_threshold:
                    heads.append(sampled_points[i])

        # Remove heads that are too close together
        heads = self._filter_close_heads(heads)

        # If no heads found, use highest point
        if len(heads) == 0 and len(sampled_elevs) > 0:
            max_idx = np.argmax(sampled_elevs)
            heads = [sampled_points[max_idx]]
        elif len(heads) == 0:
            # Fallback: use geometric approach
            heads = [self.outline[0]]

        return heads

    def _create_cost_grid(self, b=None):
        """
        Create the cost/penalty grid following Equation 2 from the paper.

        Parameters:
        -----------
        b : float
            Elevation exponent (default: use self.params['b'])

        Returns:
        --------
        cost_grid : 2D numpy array
            Cost grid with penalty values
        """
        if b is None:
            b = self.params["b"]

        # Create distance transform from glacier edges
        mask = self._create_glacier_mask()
        dist_from_edge = ndimage.distance_transform_edt(mask) * self.resolution

        # Normalize distance
        max_dist = np.max(dist_from_edge)
        if max_dist > 0:
            norm_dist = (max_dist - dist_from_edge) / max_dist
        else:
            norm_dist = np.ones_like(dist_from_edge)

        # Normalize elevation - handle case where glacier mask doesn't overlap DEM
        glacier_dem = np.where(mask, self.dem, np.nan)
        valid_elevs = glacier_dem[~np.isnan(glacier_dem)]
        
        if len(valid_elevs) > 0:
            min_elev = np.min(valid_elevs)
            max_elev = np.max(valid_elevs)
            
            if max_elev > min_elev:
                norm_elev = np.where(mask, (self.dem - min_elev) / (max_elev - min_elev), 0)
            else:
                norm_elev = np.where(mask, 0, 0)
        else:
            # No valid elevations - use distance only
            norm_elev = np.zeros_like(glacier_dem)
            norm_elev = np.where(mask, 0, 0)

        # Apply Equation 2 from the paper
        a = self.params["a"]
        f1 = self.params["f1"]
        f2 = self.params["f2"]

        cost_grid = (norm_dist**a) * f1 + (norm_elev**b) * f2

        # Set infinite cost outside glacier
        cost_grid = np.where(mask, cost_grid, np.inf)

        return cost_grid

    def _compute_least_cost_route(self, start, end, cost_grid):
        """
        Compute the least-cost route from start to end using the cost grid.

        Parameters:
        -----------
        start : tuple
            Starting point (x, y)
        end : tuple
            Ending point (x, y)
        cost_grid : 2D numpy array
            Cost grid

        Returns:
        --------
        route : array
            N x 2 array of route coordinates
        """
        # Convert coordinates to grid indices
        start_idx = (int(start[1] / self.resolution), int(start[0] / self.resolution))
        end_idx = (int(end[1] / self.resolution), int(end[0] / self.resolution))

        # Use MCP (Minimum Cost Path) from skimage
        mcp = graph.MCP(cost_grid, fully_connected=True)

        try:
            cumulative_cost, traceback = mcp.find_costs([start_idx], [end_idx])
            path = mcp.traceback(end_idx)

            # Convert back to coordinates
            route = np.array(
                [(p[1] * self.resolution, p[0] * self.resolution) for p in path]
            )

            # Smooth the route
            route = self._smooth_centerline(route)

            return route

        except:
            # If pathfinding fails, return straight line
            return np.array([[start[0], start[1]], [end[0], end[1]]], dtype=float)

    def _optimize_centerline(self, centerline, initial_cost_grid):
        """
        Optimize centerline by varying the elevation exponent b.

        Parameters:
        -----------
        centerline : array
            Initial centerline coordinates
        initial_cost_grid : 2D numpy array
            Initial cost grid

        Returns:
        --------
        optimized : array
            Optimized centerline coordinates
        """
        # Calculate upslope flow metrics
        delta_z_up, n_up = self._calculate_upslope_metrics(centerline)

        # Calculate optimization threshold (Equation 7)
        j1, j2, j3 = self.params["j1"], self.params["j2"], self.params["j3"]
        m = j1 * n_up + j2 * (delta_z_up**j3)

        # If no optimization needed
        if m == 0:
            return centerline

        # Try different b values
        best_centerline = centerline
        best_metric = delta_z_up

        b_init = self.params["b"]
        delta_b_max = 0.5

        for delta_b in np.arange(0.1, min(m, delta_b_max) + 0.1, 0.1):
            b_new = b_init + delta_b

            # Create new cost grid
            new_cost_grid = self._create_cost_grid(b=b_new)

            # Compute new route
            start = centerline[0]
            end = centerline[-1]
            new_centerline = self._compute_least_cost_route(start, end, new_cost_grid)

            # Check if improved
            new_delta_z, new_n = self._calculate_upslope_metrics(new_centerline)

            if new_delta_z < best_metric:
                best_centerline = new_centerline
                best_metric = new_delta_z

                # Check if optimization complete
                new_m = j1 * new_n + j2 * (new_delta_z**j3)
                if delta_b >= new_m:
                    break

        return best_centerline

    def _derive_branches(self, centerlines):
        """
        Split centerlines into branches and assign branch order.

        Parameters:
        -----------
        centerlines : list of arrays
            List of centerline coordinates

        Returns:
        --------
        branches : list of dict
            Branch information including order
        """
        if len(centerlines) == 0:
            return []

        # Find longest centerline (main branch)
        lengths = [self._compute_line_length(cl) for cl in centerlines]
        main_idx = np.argmax(lengths)

        branches = []

        # Main branch
        branches.append(
            {
                "coords": centerlines[main_idx],
                "order": len(centerlines),  # Highest order
                "length": lengths[main_idx],
                "is_main": True,
            }
        )

        # Process other branches
        glacier_area = self.glacier_poly.area
        k = self._compute_branch_buffer(glacier_area)

        for i, cl in enumerate(centerlines):
            if i == main_idx:
                continue

            # Trim branch where it meets main branch
            trimmed = self._trim_branch(cl, centerlines[main_idx], k)

            if trimmed is not None and len(trimmed) > 1:
                branches.append(
                    {
                        "coords": trimmed,
                        "order": 1,  # Simple order assignment
                        "length": self._compute_line_length(trimmed),
                        "is_main": False,
                    }
                )

        return branches

    # Helper methods

    def _create_glacier_mask(self):
        """Create a binary mask for the glacier area."""
        # Create grid
        y_size = self.dem.shape[0]
        x_size = self.dem.shape[1]

        mask = np.zeros((y_size, x_size), dtype=bool)

        # Simple rasterization of polygon
        for i in range(y_size):
            for j in range(x_size):
                point = Point(j * self.resolution, i * self.resolution)
                if self.glacier_poly.contains(point):
                    mask[i, j] = True

        return mask

    def _fill_depressions(self, dem):
        """Fill depressions in DEM."""
        # Simple depression filling using morphological reconstruction
        seed = np.copy(dem)
        seed[1:-1, 1:-1] = dem.max()

        filled = ndimage.grey_erosion(seed, size=(3, 3))
        filled = np.maximum(filled, dem)

        return filled

    def _compute_outline_length(self):
        """Compute total length of glacier outline."""
        total_length = 0
        for i in range(len(self.outline)):
            j = (i + 1) % len(self.outline)
            dist = np.linalg.norm(self.outline[j] - self.outline[i])
            total_length += dist
        return total_length

    def _compute_line_length(self, line):
        """Compute length of a line."""
        if len(line) < 2:
            return 0
        return np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))

    def _filter_close_heads(self, heads):
        """Remove heads that are too close together."""
        if len(heads) <= 1:
            return heads

        # Calculate minimum separation distance (Equation 1)
        glacier_area = self.glacier_poly.area
        q1, q2, rmax = self.params["q1"], self.params["q2"], self.params["rmax"]

        r = q1 * glacier_area + q2
        r = min(r, rmax)

        # Filter heads
        filtered = []
        used = set()

        heads_array = np.array(heads)
        elevs = [
            self.dem[int(h[1] / self.resolution), int(h[0] / self.resolution)]
            for h in heads
        ]

        # Sort by elevation (highest first)
        sorted_idx = np.argsort(elevs)[::-1]

        for idx in sorted_idx:
            if idx in used:
                continue

            filtered.append(heads[idx])
            used.add(idx)

            # Mark nearby heads as used
            for j in range(len(heads)):
                if j != idx and j not in used:
                    dist = np.linalg.norm(heads_array[idx] - heads_array[j])
                    if dist < r:
                        used.add(j)

        return filtered

    def _smooth_centerline(self, line):
        """Smooth centerline using PAEK algorithm approximation."""
        if len(line) < 4:
            return line

        # Calculate smoothing length (Equation 4)
        glacier_area = self.glacier_poly.area
        u1, u2, lmax = self.params["u1"], self.params["u2"], self.params["lmax"]

        l = u1 * glacier_area + u2
        l = min(l, lmax)

        # Simple smoothing using spline interpolation
        try:
            # Parametric spline interpolation
            tck, u = splprep([line[:, 0], line[:, 1]], s=l, k=min(3, len(line) - 1))

            # Evaluate spline at regular intervals
            u_new = np.linspace(0, 1, len(line))
            x_new, y_new = splev(u_new, tck)

            return np.column_stack([x_new, y_new])
        except:
            # If smoothing fails, return original
            return line

    def _calculate_upslope_metrics(self, centerline):
        """
        Calculate upslope flow metrics for optimization.

        Returns:
        --------
        delta_z_up : float
            Total elevation increase along centerline
        n_up : int
            Maximum number of consecutive upslope samples
        """
        if len(centerline) < 2:
            return 0, 0

        # Sample upper 25% of centerline
        n_samples = max(int(len(centerline) * 0.25), 2)
        upper_section = centerline[:n_samples]

        # Get elevations
        elevs = []
        for point in upper_section:
            grid_x = int(point[0] / self.resolution)
            grid_y = int(point[1] / self.resolution)

            if 0 <= grid_x < self.dem.shape[1] and 0 <= grid_y < self.dem.shape[0]:
                elevs.append(self.dem[grid_y, grid_x])

        if len(elevs) < 2:
            return 0, 0

        elevs = np.array(elevs)

        # Calculate metrics
        diffs = np.diff(elevs)
        delta_z_up = np.sum(np.maximum(diffs, 0))

        # Count consecutive upslope
        n_up = 0
        current_run = 0
        for d in diffs:
            if d > 0:
                current_run += 1
                n_up = max(n_up, current_run)
            else:
                current_run = 0

        return delta_z_up, n_up

    def _compute_branch_buffer(self, glacier_area):
        """Compute buffer width for branch separation (Equation 8)."""
        w1, w2, kmax = self.params["w1"], self.params["w2"], self.params["kmax"]

        k = w1 * glacier_area + w2
        k = min(k, kmax)

        return k

    def _trim_branch(self, branch, main_branch, buffer_dist):
        """
        Trim branch where it meets the main branch.

        Parameters:
        -----------
        branch : array
            Branch centerline to trim
        main_branch : array
            Main branch centerline
        buffer_dist : float
            Buffer distance for intersection

        Returns:
        --------
        trimmed : array or None
            Trimmed branch or None if too short
        """
        # Find where branch gets close to main branch
        min_dist_idx = None
        min_dist = np.inf

        for i, point in enumerate(branch):
            dists = np.linalg.norm(main_branch - point, axis=1)
            curr_min = np.min(dists)

            if curr_min < buffer_dist:
                if curr_min < min_dist:
                    min_dist = curr_min
                    min_dist_idx = i

        if min_dist_idx is not None and min_dist_idx > 0:
            return branch[:min_dist_idx]

        return branch


# Visualization functions
def visualize_centerline_extraction(extractor, centerlines, branches, dem_array=None, 
                                   save_path=None, figsize=(15, 10)):
    """
    Create a comprehensive visualization of the centerline extraction process.
    
    Parameters:
    -----------
    extractor : GlacierCenterlineExtractor
        The centerline extractor object
    centerlines : list of arrays
        List of centerline coordinates
    branches : list of dict
        Branch information including order and properties
    dem_array : 2D numpy array, optional
        DEM for background visualization
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize, layout='constrained')
    
    # Create mosaic layout
    mosaic = [
        ['dem', 'cost'],
        ['heads', 'final']
    ]
    axes = fig.subplot_mosaic(mosaic)
    
    # Get DEM extent from glacier outline coordinates
    if dem_array is None:
        dem_array = extractor.dem
    
    # Calculate extent from glacier outline coordinates (real coordinate system)
    outline_coords = extractor.outline
    # Handle 2D or 3D coordinates (take only first 2 dimensions)
    coords_2d = outline_coords[:, :2] if outline_coords.shape[1] > 2 else outline_coords
    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    
    # Add buffer around glacier outline for better visualization
    buffer = max((x_max - x_min), (y_max - y_min)) * 0.1
    extent = [x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer]
    
    # Plot 1: DEM with glacier outline
    im1 = axes['dem'].imshow(dem_array, cmap='terrain', extent=extent, origin='lower')
    _plot_glacier_outline(axes['dem'], extractor.outline)
    axes['dem'].set_title('DEM and Glacier Outline')
    axes['dem'].set_xlabel('Easting (m)')
    axes['dem'].set_ylabel('Northing (m)')
    axes['dem'].set_aspect('equal')
    plt.colorbar(im1, ax=axes['dem'], label='Elevation (m)', shrink=0.8)
    
    # Plot 2: Cost grid
    cost_grid = extractor._create_cost_grid()
    masked_cost = np.where(np.isinf(cost_grid), np.nan, cost_grid)
    im2 = axes['cost'].imshow(masked_cost, cmap='viridis_r', extent=extent, origin='lower')
    _plot_glacier_outline(axes['cost'], extractor.outline)
    axes['cost'].set_title('Cost Grid for Centerline Routing')
    axes['cost'].set_xlabel('Easting (m)')
    axes['cost'].set_ylabel('Northing (m)')
    axes['cost'].set_aspect('equal')
    plt.colorbar(im2, ax=axes['cost'], label='Cost Value', shrink=0.8)
    
    # Plot 3: Heads and terminus identification
    axes['heads'].imshow(dem_array, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
    _plot_glacier_outline(axes['heads'], extractor.outline)
    
    # Find and plot heads and terminus
    terminus = extractor._find_terminus()
    heads = extractor._find_heads()
    
    # Convert terminus from grid indices to real world coordinates
    # Grid coordinates are (row, col), we need (x, y) in real world coordinates
    # extent = [x_min, x_max, y_min, y_max]
    x_min, x_max, y_min, y_max = extent
    dem_height, dem_width = dem_array.shape
    
    # Convert grid indices to real coordinates
    terminus_x = x_min + terminus[1] * (x_max - x_min) / dem_width
    terminus_y = y_max - terminus[0] * (y_max - y_min) / dem_height  # Flip Y because image origin is top-left
    
    axes['heads'].plot(terminus_x, terminus_y, 'ro', markersize=10, 
                      label='Terminus', markeredgecolor='white', markeredgewidth=2)
    
    # Plot heads (heads should already be in real coordinates)
    for i, head in enumerate(heads):
        axes['heads'].plot(head[0], head[1], 'g^', markersize=10, 
                          label='Heads' if i == 0 else '', markeredgecolor='white', markeredgewidth=2)
    
    axes['heads'].set_title('Glacier Heads and Terminus')
    axes['heads'].set_xlabel('Easting (m)')
    axes['heads'].set_ylabel('Northing (m)')
    axes['heads'].set_aspect('equal')
    axes['heads'].legend()
    
    # Plot 4: Final centerlines and branches
    axes['final'].imshow(dem_array, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
    _plot_glacier_outline(axes['final'], extractor.outline)
    
    # Plot centerlines with different colors for different orders
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(branches), 1)))
    
    for i, branch in enumerate(branches):
        coords = branch['coords']
        if len(coords) > 1:
            label = f"Order {branch['order']}" + (" (Main)" if branch.get('is_main', False) else "")
            axes['final'].plot(coords[:, 0], coords[:, 1], 
                              color=colors[i % len(colors)], 
                              linewidth=3 if branch.get('is_main', False) else 2,
                              label=label)
    
    axes['final'].set_title('Extracted Centerlines')
    axes['final'].set_xlabel('Easting (m)')
    axes['final'].set_ylabel('Northing (m)')
    axes['final'].set_aspect('equal')
    if branches:
        axes['final'].legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def visualize_centerline_steps(extractor, save_dir=None, figsize=(12, 8)):
    """
    Create step-by-step visualization of the centerline extraction algorithm.
    
    Parameters:
    -----------
    extractor : GlacierCenterlineExtractor
        The centerline extractor object
    save_dir : str, optional
        Directory to save individual step figures
    figsize : tuple
        Figure size for each step
        
    Returns:
    --------
    figs : list of matplotlib Figures
        List of figures for each step
    """
    figs = []
    
    # Calculate extent from glacier outline coordinates (real coordinate system)
    outline_coords = extractor.outline
    # Handle 2D or 3D coordinates (take only first 2 dimensions)
    coords_2d = outline_coords[:, :2] if outline_coords.shape[1] > 2 else outline_coords
    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    
    # Add buffer around glacier outline for better visualization
    buffer = max((x_max - x_min), (y_max - y_min)) * 0.1
    extent = [x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer]
    
    # Step 1: Glacier outline and DEM
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    im1 = ax1.imshow(extractor.dem, cmap='terrain', extent=extent, origin='lower')
    _plot_glacier_outline(ax1, extractor.outline)
    ax1.set_title('Step 1: Glacier Outline and DEM')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    figs.append(fig1)
    
    if save_dir:
        fig1.savefig(f"{save_dir}/step1_outline_dem.png", dpi=300, bbox_inches='tight')
    
    # Step 2: Head and terminus identification
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.imshow(extractor.dem, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
    _plot_glacier_outline(ax2, extractor.outline)
    
    terminus = extractor._find_terminus()
    heads = extractor._find_heads()
    
    # Convert terminus from grid indices to real world coordinates
    x_min, x_max, y_min, y_max = extent
    dem_height, dem_width = extractor.dem.shape
    
    terminus_x = x_min + terminus[1] * (x_max - x_min) / dem_width
    terminus_y = y_max - terminus[0] * (y_max - y_min) / dem_height
    
    ax2.plot(terminus_x, terminus_y, 'ro', markersize=12, 
             label='Terminus', markeredgecolor='white', markeredgewidth=2)
    
    for i, head in enumerate(heads):
        ax2.plot(head[0], head[1], 'g^', markersize=12, 
                 label='Heads' if i == 0 else '', markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_title('Step 2: Head and Terminus Identification')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.set_aspect('equal')
    ax2.legend()
    figs.append(fig2)
    
    if save_dir:
        fig2.savefig(f"{save_dir}/step2_heads_terminus.png", dpi=300, bbox_inches='tight')
    
    # Step 3: Cost grid
    fig3, ax3 = plt.subplots(figsize=figsize)
    cost_grid = extractor._create_cost_grid()
    masked_cost = np.where(np.isinf(cost_grid), np.nan, cost_grid)
    
    im3 = ax3.imshow(masked_cost, cmap='viridis_r', extent=extent, origin='lower')
    _plot_glacier_outline(ax3, extractor.outline)
    ax3.set_title('Step 3: Cost Grid for Routing')
    ax3.set_xlabel('Easting (m)')
    ax3.set_ylabel('Northing (m)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='Cost Value')
    figs.append(fig3)
    
    if save_dir:
        fig3.savefig(f"{save_dir}/step3_cost_grid.png", dpi=300, bbox_inches='tight')
    
    # Step 4: Centerline routing
    fig4, ax4 = plt.subplots(figsize=figsize)
    ax4.imshow(extractor.dem, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
    _plot_glacier_outline(ax4, extractor.outline)
    
    # Convert terminus for routing (reuse from step 2)
    terminus_coord = (terminus_x, terminus_y)
    
    # Show routing for each head
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, head in enumerate(heads):
        try:
            centerline = extractor._compute_least_cost_route(head, terminus_coord, cost_grid)
            if len(centerline) > 1:
                ax4.plot(centerline[:, 0], centerline[:, 1], 
                        color=colors[i % len(colors)], linewidth=2, 
                        label=f'Route from Head {i+1}')
        except:
            continue
    
    ax4.plot(terminus_coord[0], terminus_coord[1], 'ko', markersize=8, 
             markeredgecolor='white', markeredgewidth=2)
    for i, head in enumerate(heads):
        ax4.plot(head[0], head[1], 'k^', markersize=8, 
                markeredgecolor='white', markeredgewidth=2)
    
    ax4.set_title('Step 4: Least-Cost Routing')
    ax4.set_xlabel('Easting (m)')
    ax4.set_ylabel('Northing (m)')
    ax4.set_aspect('equal')
    if len(heads) > 0:
        ax4.legend()
    figs.append(fig4)
    
    if save_dir:
        fig4.savefig(f"{save_dir}/step4_routing.png", dpi=300, bbox_inches='tight')
    
    return figs


def _plot_glacier_outline(ax, outline_coords, **kwargs):
    """Helper function to plot glacier outline."""
    default_kwargs = {'color': 'black', 'linewidth': 2, 'linestyle': '-'}
    default_kwargs.update(kwargs)
    
    # Close the outline if not already closed
    if not np.allclose(outline_coords[0], outline_coords[-1]):
        closed_outline = np.vstack([outline_coords, outline_coords[0]])
    else:
        closed_outline = outline_coords
    
    ax.plot(closed_outline[:, 0], closed_outline[:, 1], **default_kwargs)


def plot_centerline_analysis(centerlines, branches, glacier_outline, dem_array=None,
                           resolution=10, save_path=None, figsize=(12, 8)):
    """
    Create an analysis plot showing centerline statistics and branch hierarchy.
    
    Parameters:
    -----------
    centerlines : list of arrays
        List of centerline coordinates
    branches : list of dict
        Branch information including order and properties
    glacier_outline : array-like
        N x 2 array of glacier outline coordinates
    dem_array : 2D numpy array, optional
        DEM for background
    resolution : float
        Grid resolution in meters
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate extent from glacier outline coordinates (real coordinate system)
    glacier_coords = np.array(glacier_outline)
    # Handle 2D or 3D coordinates (take only first 2 dimensions)
    coords_2d = glacier_coords[:, :2] if glacier_coords.shape[1] > 2 else glacier_coords
    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    
    # Add buffer around glacier outline for better visualization
    buffer = max((x_max - x_min), (y_max - y_min)) * 0.1
    extent = [x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer]
    
    # Plot 1: Centerlines with branch order coloring
    if dem_array is not None:
        ax1.imshow(dem_array, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
    
    _plot_glacier_outline(ax1, glacier_outline, color='black', linewidth=2)
    
    # Color branches by order
    if branches:
        max_order = max(b['order'] for b in branches)
        colors = plt.cm.viridis(np.linspace(0, 1, max_order))
        
        for branch in branches:
            coords = branch['coords']
            if len(coords) > 1:
                color = colors[branch['order'] - 1] if branch['order'] <= max_order else 'gray'
                linewidth = 4 if branch.get('is_main', False) else 2
                linestyle = '-' if branch.get('is_main', False) else '--'
                
                ax1.plot(coords[:, 0], coords[:, 1], 
                        color=color, linewidth=linewidth, linestyle=linestyle,
                        label=f"Order {branch['order']}" + (" (Main)" if branch.get('is_main', False) else ""))
    
    ax1.set_title('Centerlines by Branch Order')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Plot 2: Branch statistics
    if branches:
        orders = [b['order'] for b in branches]
        lengths = [b['length'] for b in branches]
        is_main = [b.get('is_main', False) for b in branches]
        
        # Bar plot of branch lengths
        colors_bar = ['red' if main else 'blue' for main in is_main]
        bars = ax2.bar(range(len(branches)), lengths, color=colors_bar, alpha=0.7)
        
        # Add order labels
        for i, (order, main) in enumerate(zip(orders, is_main)):
            label = f"O{order}" + ("*" if main else "")
            ax2.text(i, lengths[i] + max(lengths) * 0.01, label, 
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_title('Branch Length Analysis')
        ax2.set_xlabel('Branch Index')
        ax2.set_ylabel('Length (m)')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Main Branch'),
                          Patch(facecolor='blue', alpha=0.7, label='Tributary')]
        ax2.legend(handles=legend_elements)
    else:
        ax2.text(0.5, 0.5, 'No branches found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Branch Length Analysis')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage function
def extract_glacier_centerlines(outline_coords, dem_array, resolution=10):
    """
    Extract centerlines for a glacier.

    Parameters:
    -----------
    outline_coords : array-like
        N x 2 array of (x, y) coordinates defining the glacier outline
    dem_array : 2D numpy array
        Digital elevation model covering the glacier area
    resolution : float
        Grid cell size in meters

    Returns:
    --------
    centerlines : list of arrays
        List of centerline coordinates
    branches : list of dict
        Branch information including order and properties
    """
    extractor = GlacierCenterlineExtractor(outline_coords, dem_array, resolution)
    centerlines, branches = extractor.extract_centerlines()

    return centerlines, branches


def extract_glacier_centerlines_with_extractor(outline_coords, dem_array, resolution=10):
    """
    Extract centerlines for a glacier and return the extractor object.
    
    Parameters:
    -----------
    outline_coords : array-like
        N x 2 array of (x, y) coordinates defining the glacier outline
    dem_array : 2D numpy array
        Digital elevation model covering the glacier area
    resolution : float
        Grid cell size in meters
        
    Returns:
    --------
    centerlines : list of arrays
        List of centerline coordinates
    branches : list of dict
        Branch information including order and properties
    extractor : GlacierCenterlineExtractor
        The extractor object for visualization
    """
    extractor = GlacierCenterlineExtractor(outline_coords, dem_array, resolution)
    centerlines, branches = extractor.extract_centerlines()
    return centerlines, branches, extractor


def create_centerline_visualizations(extractor, centerlines, branches, dem_array=None,
                                   save_path=None, create_steps=False, figsize=(15, 10)):
    """
    Create comprehensive visualizations for extracted centerlines.
    
    Parameters:
    -----------
    extractor : GlacierCenterlineExtractor
        The centerline extractor object
    centerlines : list of arrays
        List of centerline coordinates
    branches : list of dict
        Branch information including order and properties
    dem_array : 2D numpy array, optional
        DEM for background visualization
    save_path : str, optional
        Path to save the main visualization
    create_steps : bool
        Whether to create step-by-step visualizations
    figsize : tuple
        Figure size for main visualization
        
    Returns:
    --------
    results : dict
        Dictionary containing figures and visualization results
    """
    # Create main visualization
    main_fig = visualize_centerline_extraction(
        extractor, centerlines, branches, dem_array, save_path, figsize
    )
    
    results = {
        'main_figure': main_fig
    }
    
    # Create step-by-step visualizations if requested
    if create_steps:
        step_figs = visualize_centerline_steps(extractor, figsize=figsize)
        results['step_figures'] = step_figs
    
    return results
