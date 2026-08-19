"""
run.py

Download and visualize SRTM DEM for Wolverine Glacier, Alaska.

This script demonstrates:
1. Using get_rgi() to find Wolverine Glacier geometry
2. Using get_glacier_bounding_box() to create buffered bounds
3. Downloading SRTM DEM data with download_srtm_dem()
4. Loading and visualizing the DEM with xarray and matplotlib
5. Performing catchment delineation using pysheds via delineate_catchment()

Author: drotto
"""

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from pyproj import CRS

# Import flowline components
from flowline.data import get_rgi, get_glacier_bounding_box, download_srtm_dem
from flowline.topo import delineate_catchment
from flowline.centerlines import extract_glacier_centerlines_with_extractor, create_centerline_visualizations


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define output directory
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    # Get Wolverine Glacier from RGI
    print("Retrieving Wolverine Glacier geometry from RGI...")
    rgi_id = "RGI2000-v7.0-G-01-11350"  # Wolverine Glacier, Alaska
    try:
        glacier_gdf = get_rgi([rgi_id])
        glacier_row = glacier_gdf.iloc[0:1]  # Keep as GeoDataFrame
        print(
            f"Found glacier: {glacier_gdf.iloc[0].get('glac_name', 'Wolverine Glacier')}"
        )
    except Exception as e:
        print(f"Error retrieving glacier data: {e}")
        print("This may be due to missing RGI data files.")
        return

    # Get bounding box with 1000m buffer
    print("Calculating buffered bounding box...")
    buffer_m = 2000  # 1 km buffer
    lat_min, lat_max, lon_min, lon_max = get_glacier_bounding_box(glacier_row, buffer_m)
    print(
        f"Bounding box: lat({lat_min:.4f}, {lat_max:.4f}), lon({lon_min:.4f}, {lon_max:.4f})"
    )

    # Download SRTM DEM
    print("Downloading SRTM DEM...")
    try:
        dem_path = download_srtm_dem(
            lat_min, lat_max, lon_min, lon_max, output_dir=output_dir
        )
        print(f"DEM downloaded to: {dem_path}")
    except Exception as e:
        print(f"Error downloading DEM: {e}")
        print("This may be due to missing API key or network issues.")
        return

    # Load DEM with rioxarray
    print("Loading DEM with rioxarray...")
    dem_da = rxr.open_rasterio(dem_path, mask_and_scale=True)
    dem_da = dem_da.squeeze("band", drop=True)  # Remove band dimension
    print(f"Original DEM CRS: {dem_da.rio.crs}")
    print(f"Original DEM shape: {dem_da.shape}")

    # Get glacier center point for local mercator projection
    glacier_geom = glacier_row.geometry.iloc[0]
    glacier_centroid = glacier_geom.centroid
    center_lon, center_lat = glacier_centroid.x, glacier_centroid.y
    print(f"Glacier center: ({center_lon:.4f}, {center_lat:.4f})")

    # Define local mercator projection centered on glacier
    # Using UTM-like projection with custom central meridian
    local_mercator_proj = f"+proj=merc +lon_0={center_lon} +lat_ts={center_lat} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    local_crs = CRS.from_proj4(local_mercator_proj)
    print(f"Local mercator CRS: {local_crs}")

    # Reproject DEM to local mercator
    print("Reprojecting DEM to local mercator...")
    dem_da_reproj = dem_da.rio.reproject(
        local_crs,
        resolution=30,  # 30m resolution (approximately SRTM native)
        resampling=1,  # Bilinear resampling
    )
    print(f"Reprojected DEM shape: {dem_da_reproj.shape}")
    print(f"Reprojected DEM CRS: {dem_da_reproj.rio.crs}")

    # Save reprojected DEM for pysheds
    reproj_dem_path = output_dir / "wolverine_glacier_dem_reproj.tif"
    dem_da_reproj.rio.to_raster(reproj_dem_path)
    print(f"Reprojected DEM saved to: {reproj_dem_path}")

    # Use original geographic DEM for visualization
    dem_da_viz = dem_da

    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot DEM as contour plot (using original geographic coordinates)
    dem_da_viz.plot.contourf(
        ax=ax,
        levels=20,
        cmap="terrain",
        add_colorbar=True,
        cbar_kwargs={"label": "Elevation (m)"},
    )

    # Overlay glacier outline
    glacier_row.boundary.plot(
        ax=ax, color="red", linewidth=2, label="Wolverine Glacier"
    )

    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("SRTM DEM: Wolverine Glacier, Alaska")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    plot_path = output_dir / "wolverine_glacier_dem.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # Print summary statistics
    print("\n--- DEM Statistics ---")
    print("Original DEM:")
    print(f"  Min elevation: {dem_da_viz.min().values:.1f} m")
    print(f"  Max elevation: {dem_da_viz.max().values:.1f} m")
    print(f"  Mean elevation: {dem_da_viz.mean().values:.1f} m")
    print(f"  DEM shape: {dem_da_viz.shape}")
    print(f"  DEM CRS: {dem_da_viz.rio.crs}")
    print("Reprojected DEM:")
    print(f"  Min elevation: {dem_da_reproj.min().values:.1f} m")
    print(f"  Max elevation: {dem_da_reproj.max().values:.1f} m")
    print(f"  Mean elevation: {dem_da_reproj.mean().values:.1f} m")
    print(f"  DEM shape: {dem_da_reproj.shape}")
    print(f"  DEM CRS: {dem_da_reproj.rio.crs}")
    print("----------------------")

    # Perform catchment delineation using pysheds
    print("\n--- Catchment Delineation ---")
    print("Running catchment analysis using pysheds...")

    # Apply catchment delineation using the reprojected DEM
    # Specify pour point in local mercator coordinates (easting, northing)
    # 500m easting, approximately 4,181,000m northing (adjust as needed)
    pour_point_local = (500, 4180000)  # Adjust northing value as needed

    results = delineate_catchment(
        reproj_dem_path,
        pour_point_coords=pour_point_local,  # Use local mercator coordinates
        snap_threshold=1000,  # Will snap to nearest high accumulation point
        plot=False,  # We'll handle plotting with glacier overlay
    )

    # Reproject glacier geometry to local mercator for overlay
    glacier_reproj = glacier_row.to_crs(local_crs)

    # Create catchment analysis plot with glacier overlay
    from flowline.topo import _plot_catchment_analysis_with_glacier

    _plot_catchment_analysis_with_glacier(results, results["grid"], glacier_reproj)

    # Save catchment analysis plot
    catchment_plot_path = output_dir / "wolverine_catchment_analysis.png"
    plt.savefig(catchment_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Catchment analysis plot saved to: {catchment_plot_path}")

    # Print catchment statistics
    catchment_mask = results["catchment"]
    catchment_area_cells = np.sum(catchment_mask)
    print(f"Pour point coordinates: {results['pour_point']}")
    print(f"Catchment area: {catchment_area_cells} cells")
    print(f"Max flow accumulation: {np.max(results['flow_accumulation']):.0f}")
    print("Catchment delineation completed successfully!")

    print("--------------------------------")

    # Perform glacier centerline extraction
    print("\n--- Glacier Centerline Extraction ---")
    print("Extracting glacier centerlines using Kienholz et al. (2014) algorithm...")
    try:
        # Convert glacier geometry to local mercator coordinates for centerline extraction
        glacier_reproj = glacier_row.to_crs(local_crs)
        glacier_outline_coords = np.array(glacier_reproj.geometry.iloc[0].exterior.coords)
        
        # Convert DEM to numpy array
        dem_array = dem_da_reproj.values
        dem_resolution = abs(dem_da_reproj.rio.resolution()[0])  # Get actual resolution
        
        # Extract centerlines
        print(f"DEM shape: {dem_array.shape}, resolution: {dem_resolution:.1f}m")
        print("Running centerline extraction algorithm...")
        
        centerlines, branches, extractor = extract_glacier_centerlines_with_extractor(
            glacier_outline_coords, 
            dem_array, 
            resolution=dem_resolution
        )
        
        # Create visualizations
        print("Creating centerline visualizations...")
        vis_results = create_centerline_visualizations(
            extractor, centerlines, branches, dem_array,
            save_path=output_dir / "wolverine_centerlines.png",
            create_steps=True,
            figsize=(16, 12)
        )
        
        # Print centerline statistics
        print("\n--- Centerline Results ---")
        print(f"Number of centerlines extracted: {len(centerlines)}")
        print(f"Number of branches identified: {len(branches)}")
        
        if branches:
            for i, branch in enumerate(branches):
                branch_type = "Main" if branch.get('is_main', False) else "Tributary"
                print(f"  Branch {i+1}: Order {branch['order']}, Length {branch['length']:.0f}m ({branch_type})")
        
        # Save step-by-step visualizations
        if 'step_figures' in vis_results:
            step_figures = vis_results['step_figures']
            step_dir = output_dir / "centerline_steps"
            step_dir.mkdir(exist_ok=True)
            
            step_names = ['outline_dem', 'heads_terminus', 'cost_grid', 'routing']
            for i, (fig, name) in enumerate(zip(step_figures, step_names)):
                step_path = step_dir / f"step{i+1}_{name}.png"
                fig.savefig(step_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            print(f"Step-by-step visualizations saved to: {step_dir}")
        
        print(f"Main centerline visualization saved to: {output_dir / 'wolverine_centerlines.png'}")
        print("Centerline extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during centerline extraction: {e}")
        print("This may be due to complex glacier geometry or DEM processing issues.")
        import traceback
        traceback.print_exc()

    print("-----------------------------------")


if __name__ == "__main__":
    main()
