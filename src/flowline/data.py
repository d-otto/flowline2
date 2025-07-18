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
import sqlite3 as sq

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

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
                #rgi = rgi.loc[rgi.rgi_id.isin(rgiids)]
                concat.append(rgi)
            rgi = pd.concat(concat, ignore_index=True)
            
        elif version == 6:
            if from_sqllite:
                con = sq.connect(Path(ROOT, "data/interim/rgi.sqllite"))
                with con:  # context manager bullshit (https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/)
                    query = (
                        f"SELECT * FROM rgi WHERE RGIId in ({','.join(['?'] * len(rgiids))})"
                    )
                    rgi = pd.read_sql_query(query, con, params=rgiids)
                con.close()
            else:
                # todo: generalize
                p = Path(GDATA_DIR, "rgi60/02_rgi60_WesternCanadaUS")
                rgi = gpd.read_file(p)
            rgi = rgi.rename(columns={
                'RGIId': 'rgi_id',
                'Lmax': 'lmax_m',
                'Name': 'glac_name',
                'Area': 'area_km2',
            })
        

    #rgi["geometry"] = rgi.apply(lambda x: shapely.wkt.loads(x.geometry), axis=1)
    rgi = gpd.GeoDataFrame(rgi, crs='EPSG:4326')
    rgi = rgi.set_index('rgi_id')

    return rgi

################################################################################
def read_berk_earth(p):
    p = Path(ROOT, "data/berk_earth/42875-TAVG-Data.txt")
    df = pd.read_table(
        p,
        comment='%',
        sep=' +',
        header=None,
        on_bad_lines='warn',
        names=["year", "month", "raw_temp", "raw_temp_anom", "qc_failed", "continuity_breaks", "adj_temp", "adj_anom", "regional_exp_temp", "regional_exp_anom"],
        engine='python'
    )  # sep = regex for "at least one space"
    
    # # Fill NaN values with 1 in the 'month' column (or adjust as needed)
    # df['month'] = df['month'].fillna(1)
    
    # Convert 'year' and 'month' columns to integers
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    # Combine 'year' and 'month' columns into 'date' column
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    return df