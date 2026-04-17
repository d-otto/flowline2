# -*- coding: utf-8 -*-
"""
topo.py

Topographic analysis tools using pysheds for catchment delineation,
flow direction calculation, and flow accumulation analysis.

Author: drotto
Created: 2025-08-20
Project: flowline
"""

import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder

import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from skimage import graph
from shapely.geometry import Polygon, Point, LineString
import warnings


def delineate_catchment(
    dem_path, pour_point_coords=None, snap_threshold=1000, plot=True
):
    """
    Complete catchment delineation workflow using pysheds.

    This function performs the full workflow for catchment analysis including:
    - DEM conditioning (flat resolution)
    - Flow direction calculation using D-infinity routing
    - Flow accumulation computation
    - Flow distance calculation
    - Catchment delineation from pour point
    - Optional visualization of all results

    Parameters:
    -----------
    dem_path : str
        Path to DEM TIFF file
    pour_point_coords : tuple, optional
        (x, y) coordinates for pour point in the DEM's coordinate system.
        If None, will use the point with maximum flow accumulation.
    snap_threshold : int, default 1000
        Accumulation threshold for pour point snapping to high accumulation cells
    plot : bool, default True
        Whether to generate plots of all analysis steps

    Returns:
    --------
    dict : Results dictionary containing:
        - 'dem': Original DEM data
        - 'inflated_dem': DEM after flat resolution
        - 'flow_directions': Flow direction raster
        - 'flow_accumulation': Flow accumulation raster
        - 'flow_distance': Flow distance raster
        - 'catchment': Delineated catchment mask
        - 'pour_point': Final pour point coordinates (x, y)
        - 'raster': pysheds Raster object for further analysis

    Example:
    --------
    >>> results = delineate_catchment('dem.tif', pour_point_coords=(-120.5, 47.2))
    >>> catchment = results['catchment']
    >>> flow_acc = results['flow_accumulation']
    """

    # Load DEM using rioxarray first to handle nodata properly
    print(f"Reading DEM from: {dem_path}")

    # Load with rioxarray to get proper nodata handling
    dem_da = rxr.open_rasterio(dem_path, mask_and_scale=True)

    # Get data and metadata
    dem_data = dem_da.values[0]  # Remove band dimension
    transform = dem_da.rio.transform()
    crs = dem_da.rio.crs

    # Handle nodata value - use a standard integer value to avoid NumPy 2.0 issues
    # Use -9999 which is a common standard for elevation nodata
    nodata_value = -9999
    print(f"Setting standard nodata value: {nodata_value}")

    # Ensure data is float64 for pysheds compatibility
    dem_data = dem_data.astype(np.float64)

    # Create pysheds Grid from the rioxarray data

    viewfinder = ViewFinder(
        affine=transform, shape=dem_data.shape, crs=crs, nodata=nodata_value
    )

    grid = Grid(viewfinder=viewfinder)
    dem_raster = Raster(dem_data, viewfinder)

    print(f"Grid extent: {grid.extent}")
    print(f"Grid CRS: {grid.crs}")
    print(f"DEM raster shape: {dem_raster.shape}")

    # Get DEM data for statistics
    data = dem_raster.copy()
    print(
        f"DEM stats: min={np.min(data):.1f}, max={np.max(data):.1f}, shape={data.shape}"
    )

    # DEM conditioning - let pysheds throw errors if it fails
    print("Conditioning DEM...")
    pit_filled_dem = grid.fill_pits(dem_raster)
    print("Pit filling successful")

    flooded_dem = grid.fill_depressions(pit_filled_dem)
    print("Depression filling successful")

    inflated_dem = grid.resolve_flats(flooded_dem)
    print("Flat resolution successful")

    # Compute flow directions using D-infinity routing
    print("Computing flow directions using D-infinity routing...")
    flow_directions = grid.flowdir(inflated_dem, routing="dinf")
    print("Flow direction calculation successful")

    # Compute flow accumulation
    print("Computing flow accumulation...")
    flow_accumulation = grid.accumulation(flow_directions, routing="dinf")

    print(
        f"Flow accumulation stats: min={np.min(flow_accumulation):.1f}, max={np.max(flow_accumulation):.1f}"
    )
    print(f"Non-zero flow accumulation cells: {np.sum(flow_accumulation > 0)}")

    # Handle pour point specification
    if pour_point_coords is None:
        # Find point with maximum flow accumulation
        max_acc_idx = np.unravel_index(
            np.argmax(flow_accumulation), flow_accumulation.shape
        )
        # Convert array indices to coordinates using Grid's coordinate system
        col, row = max_acc_idx[1], max_acc_idx[0]
        x_pour = grid.affine[2] + col * grid.affine[0]
        y_pour = grid.affine[5] + row * grid.affine[4]
        print(
            f"Auto-detected pour point at max accumulation: ({x_pour:.1f}, {y_pour:.1f})"
        )
    else:
        x_pour, y_pour = pour_point_coords
        print(f"Using provided pour point: ({x_pour:.1f}, {y_pour:.1f})")

    # Snap pour point to high accumulation cell
    high_acc_mask = flow_accumulation > snap_threshold
    if np.sum(high_acc_mask) == 0:
        print(f"No cells above threshold {snap_threshold}, using original pour point")
        x_snap, y_snap = x_pour, y_pour
    else:
        x_snap, y_snap = grid.snap_to_mask(high_acc_mask, (x_pour, y_pour))

    # Delineate catchment
    print("Delineating catchment...")
    catchment = grid.catchment(
        x=x_snap, y=y_snap, fdir=flow_directions, xytype="coordinate", routing="dinf"
    )

    # Calculate flow distance to outlet
    print("Computing flow distance...")
    flow_distance = grid.distance_to_outlet(
        x=x_snap, y=y_snap, fdir=flow_directions, xytype="coordinate", routing="dinf"
    )

    # Prepare results dictionary
    results = {
        "dem": data,
        "inflated_dem": inflated_dem,
        "flow_directions": flow_directions,
        "flow_accumulation": flow_accumulation,
        "flow_distance": flow_distance,
        "catchment": catchment,
        "pour_point": (x_snap, y_snap),
        "raster": dem_raster,
        "grid": grid,
    }

    # Generate plots if requested
    if plot:
        _plot_catchment_analysis(results, grid)

    return results


def _plot_catchment_analysis(results, grid):
    """
    Create comprehensive plots of catchment analysis results.

    Parameters:
    -----------
    results : dict
        Results dictionary from delineate_catchment()
    grid : pysheds.Grid
        Grid object for coordinate system information
    """

    # Create subplot mosaic layout
    mosaic = """
    AB
    CD
    EE
    """

    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 10))

    # Map subplot labels to meaningful names
    ax_mapping = {
        "A": "dem",
        "B": "flow_dir",
        "C": "flow_acc",
        "D": "flow_dist",
        "E": "catchment",
    }

    # Rename axes for clarity
    axes_renamed = {ax_mapping[k]: v for k, v in axes.items()}

    # Get extent for consistent plotting
    extent = grid.extent

    # Use renamed axes
    axes = axes_renamed

    # Plot 1: Original DEM
    im1 = axes["dem"].imshow(results["dem"], extent=extent, cmap="terrain")
    axes["dem"].set_title("Digital Elevation Model")
    axes["dem"].set_xlabel("Easting")
    axes["dem"].set_ylabel("Northing")
    plt.colorbar(im1, ax=axes["dem"], label="Elevation (m)")

    # Plot 2: Flow Directions
    im2 = axes["flow_dir"].imshow(
        results["flow_directions"], extent=extent, cmap="viridis"
    )
    axes["flow_dir"].set_title("Flow Directions")
    axes["flow_dir"].set_xlabel("Easting")
    axes["flow_dir"].set_ylabel("Northing")
    plt.colorbar(im2, ax=axes["flow_dir"], label="Direction Code")

    # Plot 3: Flow Accumulation (log scale for better visualization)
    flow_acc_log = np.log10(results["flow_accumulation"] + 1)
    im3 = axes["flow_acc"].imshow(flow_acc_log, extent=extent, cmap="Blues")
    axes["flow_acc"].set_title("Flow Accumulation (log10)")
    axes["flow_acc"].set_xlabel("Easting")
    axes["flow_acc"].set_ylabel("Northing")
    plt.colorbar(im3, ax=axes["flow_acc"], label="log10(Accumulation + 1)")

    # Plot 4: Flow Distance
    im4 = axes["flow_dist"].imshow(
        results["flow_distance"], extent=extent, cmap="plasma"
    )
    axes["flow_dist"].set_title("Flow Distance")
    axes["flow_dist"].set_xlabel("Easting")
    axes["flow_dist"].set_ylabel("Northing")
    plt.colorbar(im4, ax=axes["flow_dist"], label="Distance")

    # Plot 5: Delineated Catchment
    catchment_masked = np.where(results["catchment"], results["catchment"], np.nan)
    axes["catchment"].imshow(catchment_masked, extent=extent, cmap="Reds", alpha=0.7)
    axes["catchment"].imshow(results["dem"], extent=extent, cmap="terrain", alpha=0.3)
    axes["catchment"].set_title("Delineated Catchment")
    axes["catchment"].set_xlabel("Easting")
    axes["catchment"].set_ylabel("Northing")

    # Mark pour point
    x_pour, y_pour = results["pour_point"]
    axes["catchment"].plot(
        x_pour,
        y_pour,
        "k*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Pour Point",
    )
    axes["catchment"].legend()

    plt.tight_layout()
    plt.show()

    return fig


def _plot_catchment_analysis_with_glacier(results, grid, glacier_gdf):
    """
    Create comprehensive plots of catchment analysis results with glacier outline overlay.

    Parameters:
    -----------
    results : dict
        Results dictionary from delineate_catchment()
    grid : pysheds.Grid
        Grid object for coordinate system information
    glacier_gdf : geopandas.GeoDataFrame
        Glacier geometry in the same CRS as the grid
    """

    # Create subplot mosaic layout
    mosaic = """
    AB
    CD
    EE
    """

    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 10))

    # Map subplot labels to meaningful names
    ax_mapping = {
        "A": "dem",
        "B": "flow_dir",
        "C": "flow_acc",
        "D": "flow_dist",
        "E": "catchment",
    }

    # Rename axes for clarity
    axes_renamed = {ax_mapping[k]: v for k, v in axes.items()}

    # Get extent for consistent plotting
    extent = grid.extent

    # Use renamed axes
    axes = axes_renamed

    # Plot 1: Original DEM
    im1 = axes["dem"].imshow(results["dem"], extent=extent, cmap="terrain")
    axes["dem"].set_title("Digital Elevation Model")
    axes["dem"].set_xlabel("Easting")
    axes["dem"].set_ylabel("Northing")
    plt.colorbar(im1, ax=axes["dem"], label="Elevation (m)")
    # Add glacier outline
    glacier_gdf.boundary.plot(ax=axes["dem"], color="black", linewidth=1)

    # Plot 2: Flow Directions
    im2 = axes["flow_dir"].imshow(
        results["flow_directions"], extent=extent, cmap="viridis"
    )
    axes["flow_dir"].set_title("Flow Directions")
    axes["flow_dir"].set_xlabel("Easting")
    axes["flow_dir"].set_ylabel("Northing")
    plt.colorbar(im2, ax=axes["flow_dir"], label="Direction Code")
    # Add glacier outline
    glacier_gdf.boundary.plot(ax=axes["flow_dir"], color="black", linewidth=1)

    # Plot 3: Flow Accumulation (log scale for better visualization)
    flow_acc_log = np.log10(results["flow_accumulation"] + 1)
    im3 = axes["flow_acc"].imshow(flow_acc_log, extent=extent, cmap="Blues")
    axes["flow_acc"].set_title("Flow Accumulation (log10)")
    axes["flow_acc"].set_xlabel("Easting")
    axes["flow_acc"].set_ylabel("Northing")
    plt.colorbar(im3, ax=axes["flow_acc"], label="log10(Accumulation + 1)")
    # Add glacier outline
    glacier_gdf.boundary.plot(ax=axes["flow_acc"], color="black", linewidth=1)

    # Plot 4: Flow Distance
    im4 = axes["flow_dist"].imshow(
        results["flow_distance"], extent=extent, cmap="plasma"
    )
    axes["flow_dist"].set_title("Flow Distance")
    axes["flow_dist"].set_xlabel("Easting")
    axes["flow_dist"].set_ylabel("Northing")
    plt.colorbar(im4, ax=axes["flow_dist"], label="Distance")
    # Add glacier outline
    glacier_gdf.boundary.plot(ax=axes["flow_dist"], color="black", linewidth=1)

    # Plot 5: Delineated Catchment
    catchment_masked = np.where(results["catchment"], results["catchment"], np.nan)
    axes["catchment"].imshow(catchment_masked, extent=extent, cmap="Reds", alpha=0.7)
    axes["catchment"].imshow(results["dem"], extent=extent, cmap="terrain", alpha=0.3)
    axes["catchment"].set_title("Delineated Catchment")
    axes["catchment"].set_xlabel("Easting")
    axes["catchment"].set_ylabel("Northing")

    # Mark pour point
    x_pour, y_pour = results["pour_point"]
    axes["catchment"].plot(
        x_pour,
        y_pour,
        "k*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Pour Point",
    )
    # Add glacier outline
    glacier_gdf.boundary.plot(
        ax=axes["catchment"], color="black", linewidth=1, label="Glacier Outline"
    )
    axes["catchment"].legend()

    plt.tight_layout()
    plt.show()

    return fig


