# -*- coding: utf-8 -*-
"""
data.py

Description.

Author: drotto
Created: 2/2/24 @ 06:40
Project: flowline
"""

from pathlib import Path
from datetime import datetime
import requests
import sqlite3 as sq
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import rasterio
import xarray as xr

from flowline.utils import ROOT, GDATA_DIR


################################################################################
def get_rgi(rgiids, from_sqllite=False, version=7):
    if not isinstance(rgiids, list):
        rgiids = [rgiids]

    if from_sqllite:
        con = sq.connect(Path(ROOT, "data/interim/rgi.sqllite"))
        with con:  # context manager bullshit (https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/)
            query = (
                f"SELECT * FROM rgi WHERE RGIId in ({','.join(['?'] * len(rgiids))})"
            )
            rgi = pd.read_sql_query(query, con, params=rgiids)
        con.close()

    else:
        if version == 7:
            p = Path(GDATA_DIR, "rgi7/RGI2000-v7.0-G")
            p = [x for x in p.iterdir() if x.is_dir()]  # todo: revise these conditions
            p = [x for x in p if not x.name.startswith("00")]
            concat = []
            for pp in p:
                print(pp)
                rgi = gpd.read_file(pp)
                concat.append(rgi)
            rgi = pd.concat(concat, ignore_index=True)
            
            # Filter to requested glacier IDs if specific IDs were provided
            if rgiids:
                rgi = rgi.loc[rgi.rgi_id.isin(rgiids)]

        elif version == 6:
            if from_sqllite:
                con = sq.connect(Path(ROOT, "data/interim/rgi.sqllite"))
                with con:  # context manager bullshit (https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/)
                    query = f"SELECT * FROM rgi WHERE RGIId in ({','.join(['?'] * len(rgiids))})"
                    rgi = pd.read_sql_query(query, con, params=rgiids)
                con.close()
            else:
                # todo: generalize
                p = Path(GDATA_DIR, "rgi60/02_rgi60_WesternCanadaUS")
                rgi = gpd.read_file(p)
            rgi = rgi.rename(
                columns={
                    "RGIId": "rgi_id",
                    "Lmax": "lmax_m",
                    "Name": "glac_name",
                    "Area": "area_km2",
                }
            )

    # rgi["geometry"] = rgi.apply(lambda x: shapely.wkt.loads(x.geometry), axis=1)
    rgi = gpd.GeoDataFrame(rgi, crs="EPSG:4326")
    rgi = rgi.set_index("rgi_id")

    return rgi


################################################################################
def read_berk_earth(p):
    p = Path(ROOT, "data/berk_earth/42875-TAVG-Data.txt")
    df = pd.read_table(
        p,
        comment="%",
        sep=" +",
        header=None,
        on_bad_lines="warn",
        names=[
            "year",
            "month",
            "raw_temp",
            "raw_temp_anom",
            "qc_failed",
            "continuity_breaks",
            "adj_temp",
            "adj_anom",
            "regional_exp_temp",
            "regional_exp_anom",
        ],
        engine="python",
    )  # sep = regex for "at least one space"

    # # Fill NaN values with 1 in the 'month' column (or adjust as needed)
    # df['month'] = df['month'].fillna(1)

    # Convert 'year' and 'month' columns to integers
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    # Combine 'year' and 'month' columns into 'date' column
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    return df


def download_srtm_dem(
    lat_min, lat_max, lon_min, lon_max, demtype="COP30", output_dir=None, force=False
):
    """
    Download SRTM DEM data from OpenTopography API for a specified bounding box.

    Parameters:
    -----------
    lat_min : float
        Minimum latitude (south bound)
    lat_max : float
        Maximum latitude (north bound)
    lon_min : float
        Minimum longitude (west bound)
    lon_max : float
        Maximum longitude (east bound)
    demtype : str, optional
        DEM dataset type. Default is "COP30" (Copernicus DEM 30m)
    output_dir : Path or str, optional
        Output directory. Default is data/external/
    force : bool, optional
        If True, re-download even if file already exists. Default is False

    Returns:
    --------
    Path
        Path to downloaded GeoTIFF file

    Example:
    --------
    >>> # Download Copernicus DEM data for Juneau Icefield area
    >>> dem_path = download_srtm_dem(58.0, 59.0, -135.0, -133.0)
    """

    try:
        from api_keys import OPENTOPOGRAPHY_API_KEY
    except ImportError:
        raise ImportError(
            "API key file not found. Please create api_keys.py with your OpenTopography API key."
        )

    # Set default output directory
    if output_dir is None:
        output_dir = ROOT / "data" / "external"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct API URL
    base_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPOGRAPHY_API_KEY,
    }

    # Generate filename
    filename = f"{demtype.lower()}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.tif"
    output_path = output_dir / filename

    # Check if file already exists
    if output_path.exists() and not force:
        logging.info(f"DEM file already exists: {output_path}")
        logging.info(f"Skipping download. Use force=True to re-download.")
        return output_path

    # Download DEM
    logging.info(f"Downloading {demtype} DEM for bounds: lat({lat_min}, {lat_max}), lon({lon_min}, {lon_max})")
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    # Save to file
    logging.info(f"Saving DEM to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(response.content)

    logging.info(f"Successfully downloaded DEM: {filename}")
    return output_path


################################################################################
def get_glacier_bounding_box(glacier_gdf, buffer):
    """
    Get bounding box coordinates for a glacier with buffer.
    
    Parameters:
    -----------
    glacier_gdf : geopandas.GeoDataFrame
        Glacier geometry from get_rgi()
    buffer : float or tuple
        Buffer distance in meters. If float, applies uniform buffer.
        If tuple (top, right, bottom, left), applies asymmetric buffer.
    
    Returns:
    --------
    tuple
        (lat_min, lat_max, lon_min, lon_max) for use with download_srtm_dem
    
    Example:
    --------
    >>> rgi = get_rgi(['RGI60-01.00570'])  # Wolverine Glacier
    >>> bounds = get_glacier_bounding_box(rgi.iloc[0:1], 1000)  # 1km buffer
    >>> lat_min, lat_max, lon_min, lon_max = bounds
    """
    
    # Convert to projected coordinate system for accurate buffer calculation
    # Use UTM zone based on longitude of glacier centroid
    glacier_centroid = glacier_gdf.geometry.centroid.iloc[0]
    lon_center = glacier_centroid.x
    
    # Determine UTM zone (simplified - works for most glaciers)
    utm_zone = int((lon_center + 180) / 6) + 1
    if glacier_centroid.y >= 0:
        utm_crs = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
    else:
        utm_crs = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere
    
    # Project to UTM for accurate buffer calculation
    glacier_utm = glacier_gdf.to_crs(utm_crs)
    
    # Apply buffer
    if isinstance(buffer, (int, float)):
        # Uniform buffer
        buffered = glacier_utm.buffer(buffer)
    else:
        # Asymmetric buffer (top, right, bottom, left)
        top, right, bottom, left = buffer
        bounds = glacier_utm.bounds.iloc[0]
        # Create buffered rectangle
        buffered_box = box(
            bounds['minx'] - left,
            bounds['miny'] - bottom, 
            bounds['maxx'] + right,
            bounds['maxy'] + top
        )
        buffered = gpd.GeoDataFrame([0], geometry=[buffered_box], crs=utm_crs)
    
    # Convert back to geographic coordinates
    if isinstance(buffered, gpd.GeoSeries):
        buffered = gpd.GeoDataFrame(geometry=buffered, crs=utm_crs)
    
    buffered_geo = buffered.to_crs("EPSG:4326")
    
    # Extract bounding box
    bounds = buffered_geo.bounds.iloc[0]
    lat_min, lat_max = bounds['miny'], bounds['maxy']
    lon_min, lon_max = bounds['minx'], bounds['maxx']
    
    return lat_min, lat_max, lon_min, lon_max
