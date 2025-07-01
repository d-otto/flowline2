"""
Glacier flowline geometry processing from OGGM

Extracts geometrical data from OGGM outputs and combines elevation bands with geometrical centerlines to create a length-accurate 2D representation of glacier geometries.

Key components:
- get_flowline_geom: Class for extracting and processing flowline geometries from OGGM outputs
- extract_flowline_geom: Function to extract flowline geometry from OGGM workflow

Usage:
    from flowline.geometry import get_flowline_geom
    
    # Initialize a flowline geometry object for a specific RGI ID
    geom = get_flowline_geom(rgiid="RGI60-11.00897", name="Hintereisferner")
    
    # Access geometry data
    x = geom.x  # horizontal distance along flowline
    zb = geom.zb  # bed elevation
    w = geom.w  # glacier width
    h0 = geom.h0  # ice thickness
    
    # Convert to pandas DataFrame
    df = geom.to_pandas()
"""

from pathlib import Path
import shutil
import os
import pickle
import gzip
import logging
from typing import Union, Optional, List, Dict, Any

import dill
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import netCDF4 as nc

# Import from configuration or environment variables
from flowline.utils import ROOT, GDATA_DIR

# Set up logging
logger = logging.getLogger(__name__)

# Configuration for paths - can be overridden
DEFAULT_FLOWLINES_DIR = Path(ROOT, 'data/oggm_flowlines/e_bands')


class get_flowline_geom:
    """
    Extracts and processes glacier flowline geometry data from OGGM outputs.
    
    This class combines OGGM geometrical centerlines with elevation bands to get 
    a length-accurate bulk 2D representation of the glacier geometry. It processes 
    bed elevation, width, and ice thickness data along the flowline.
    
    Parameters
    ----------
    rgiid : str
        The RGI (Randolph Glacier Inventory) ID of the glacier to process.
    dirname : Path, optional
        Directory containing the OGGM flowline data files.
        Defaults to DEFAULT_FLOWLINES_DIR.
    name : str, optional
        Name of the glacier, used in plot titles and filenames.
    raw : bool, optional
        If True, skips smoothing and adjustment of the geometric data.
        If False (default), applies smoothing and glacier-specific adjustments.
    quiet : bool, optional
        If True, suppresses generation of quality control plots.
        If False (default), generates QC plots.
    rgi : object, optional
        RGI glacier object containing metadata like area_km2 and lmax_m for reference
        in plots. Required if quiet=False.
        
    Attributes
    ----------
    x : numpy.ndarray
        Horizontal distance along flowline (m).
    zb : numpy.ndarray
        Bed elevation (m).
    w : numpy.ndarray
        Glacier width (m).
    h0 : numpy.ndarray
        Initial ice surface thickness (m).
    
    Notes
    -----
    There are four sources of information used:
    - Elevation bands (eb) and geometrical centerlines (gc) are the two types of models.
    - For each, there is a glacier flowline (fl; using the primary flowline for 
      geometrical centerlines) and downstream lines (dl).
    
    We obtain the following variables from the combination of these: 
    - x: horizontal distance along flowline
    - zb: bed elevation
    - w: glacier width
    - h0: initial ice surface thickness
    
    Using the main flowline of the GC, we do not include ice from tributaries that 
    comes from higher elevations. However, we take the length of this flowline as 
    the "true" length of the glacier.
    """
    
    def __init__(self, rgiid: str, 
                dirname: Path = DEFAULT_FLOWLINES_DIR, 
                name: Optional[str] = None, 
                raw: bool = False, 
                quiet: bool = False, 
                rgi: Optional[Any] = None):
        '''Combine oggm geometrical centerlines with elevation bands to get a length-accurate bulk 2d representation of the glacier geometry.

        There are four sources of information. Elevation bands (eb) and geometrical centerlines (gc) are the two types of models.
        For each, there is a glacier flowline (fl; using the primary flowline for geometrical centerlines) and downstream lines (dl).
        We need to obtain the following variables from the combination of these: x, zb, w, and h.
        Where x is horizontal length, zb is bed elevation, w is glacier width, and h is the initial ice surface thickness.

        Using the main flowline of the GC, we do not include ice from tributaries that comes from higher elevations.
        However, we take the length of this flowline as the "true" length of the glacier.

        Parameters
        ----------
        rgiid :
        dirname :
        '''
        
        # Load files created by extract_flowline_geom
        # Create paths to input files
        data_dir = Path(dirname)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            
        # Load the main flowline data
        fl_path = data_dir / f'{rgiid}.model_flowlines.pickle'
        try:
            with gzip.open(fl_path, "rb") as openfile:
                fl = dill.load(openfile)[0]
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Failed to load flowline data from {fl_path}: {e}")
            raise

        # Load the inversion output
        inv_path = data_dir / f'{rgiid}.inversion_output.pickle'
        try:
            with gzip.open(inv_path, "rb") as openfile:
                inv = dill.load(openfile)[0]
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Failed to load inversion data from {inv_path}: {e}")
            raise

        # Load the downstream line
        dsl_path = data_dir / f'{rgiid}.downstream_line.pickle'
        try:
            with gzip.open(dsl_path, "rb") as openfile:
                dsl = dill.load(openfile)[0]
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Failed to load downstream line data from {dsl_path}: {e}")
            raise

        # Load elevation band flowline if needed
        # ebfl_path = data_dir / f'{rgiid}.elevation_band_flowline.pickle'
        # with gzip.open(ebfl_path, "rb") as openfile:
        #     ebfl = dill.load(openfile)[0]
        

        # Extract basic geometric data
        line_len = len(fl.surface_h)
        x = np.arange(0, line_len) * fl.map_dx
        h0 = fl.thick
        zb = fl.bed_h
        # Width info that has been adjusted to the RGI area is only in the inversion flowline
        # There is a weird issue here where the width of the terminus goes to zero, 
        # but then the downstream line width jumps up to the real valley width.
        w = fl.widths_m
        w[:fl.terminus_index+1] = inv['width']

        # Create and save raw dataframe
        df = pd.DataFrame(dict(x=x, zb=zb, w=w, h0=h0))
        raw_output_path = data_dir / f'flowline_geom_{rgiid}_raw.csv'
        df.to_csv(raw_output_path, index=False)
        
        # Apply glacier-specific adjustments
        if rgiid == "RGI60-11.00897":  # Hintereisferner
            df['zb'].iloc[133:145] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['zb'].iloc[0:5] = np.linspace(3570, 3550, 5)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
        elif rgiid == 'RGI60-11.03638':  # argentiere
            df = df.iloc[:400]
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
        elif rgiid == 'RGI60-02.18778':  # south cascade
            # todo: compare better with fountain 1994, also earlier
            df = df.iloc[:120]

            # crude manual adjustment for SCG bedfitting experiment
            df.loc[0:100, 'zb'] = np.nan
            df.loc[0, 'zb'] = 2115
            df.loc[2, 'zb'] = 2085
            df.loc[5, 'zb'] = 1990
            df.loc[10, 'zb'] = 1960  # 500
            df.loc[15, 'zb'] = 1940  # 750
            df.loc[20, 'zb'] = 1925  # 1000
            df.loc[25, 'zb'] = 1840  # 1250
            df.loc[30, 'zb'] = 1830  # 1500
            df.loc[31, 'zb'] = 1835  # 1550
            df.loc[32, 'zb'] = 1835  # 1600
            df.loc[33, 'zb'] = 1840  # 1650
            df.loc[34, 'zb'] = 1835  # 1650
            df.loc[35, 'zb'] = 1815  # 1750
            df.loc[37, 'zb'] = 1785  # 1800
            df.loc[40, 'zb'] = 1630  # 2000
            df.loc[43, 'zb'] = 1660  # 2150
            df.loc[46, 'zb'] = 1770  # 2300
            df.loc[47, 'zb'] = 1755  # 2350
            df.loc[49, 'zb'] = 1740  # 2450
            df.loc[50, 'zb'] = 1740  # 2500
            df.loc[51, 'zb'] = 1710  # 2550
            df.loc[53, 'zb'] = 1650  # 2650
            df.loc[55, 'zb'] = 1620  # 2750
            df.loc[58, 'zb'] = 1610
            df.loc[65, 'zb'] = 1613  # 3250
            df.loc[70, 'zb'] = 1600
            df.loc[80, 'zb'] = 1610
            df.loc[90, 'zb'] = 1590
            df['zb'] = df['zb'].interpolate(method='quadratic', order=3)
            if raw is False:
                #df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = np.maximum((df['h0'] + zb[:len(df['zb'])]) - df['zb'], 0)
        elif rgiid == "RGI60-01.09162":  # wolverine
            df = df.iloc[:300]
            df['zb'].iloc[0:10] = np.linspace(1505, 1496, 10)

            df['zb'].iloc[143:159] = np.nan
            df['zb'].iloc[156] = 425
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:10] = 105
        elif rgiid == "RGI60-01.00570":  # Gulkana
            df = df.iloc[:300]
            df['zb'].iloc[0:4] = np.linspace(2350, 2310, 4)
            df['zb'].iloc[135:173] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:4] = 70
        elif rgiid == "RGI60-11.01450":  # Aletsch
            df['zb'].iloc[0:3] = np.linspace(4000, 3988, 3)
            df['h0'].iloc[0:2] = 60
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=30, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:9] = 90
        elif rgiid == "RGI60-11.01346":  # Unterer Grindelwald
            df = df.iloc[:400]
            df['zb'].iloc[160:183] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['zb'].iloc[0:4] = np.linspace(3920, 3909, 4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:2] = 80
        elif rgiid == "RGI60-11.01238":  # rhone
            df = df.iloc[:400]
            df['zb'].iloc[0:12] = np.linspace(3460, 3450, 12)
            df['zb'].iloc[175:197] = np.nan
            df['zb'].iloc[180] = 2200
            df['zb'].iloc[188] = 2205
            df['zb'].iloc[195] = 2225
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:9] = 90
        elif rgiid == "RGI60-11.03646":  # Bossons
            df = df.iloc[:300]
            df['zb'].iloc[0:11] = np.linspace(4700, 4664, 11)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
                df['h0'].iloc[0:11] = 50
        elif rgiid == "RGI60-11.03643":  # mer de glace
            df = df.iloc[:500]
            df['zb'].iloc[256:263] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
        elif rgiid == "RGI60-02.17739":  # Easton
            df = df.iloc[:250]
            if raw is False:
                df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
                df['h0'] = df['h0'] - (df['zb'] - zb[0: len(df['zb'])])
            
            
        # Smooth out the widths if not using raw data
        if raw is False:
            w0 = df['w']
            w1 = sci.signal.savgol_filter(df['w'], window_length=10, polyorder=4)
            factor = w0.sum()/w1.sum()  # conserve total area
            df['w'] = w1 * factor
            
        
        # Zero thickness after terminus
        df['h0'].iloc[np.argmin(h0):] = 0
        
        # Save processed dataframe
        output_path = data_dir / f'flowline_geom_{rgiid}.csv'
        df.to_csv(output_path, index=False)

        # Generate QC plots if not in quiet mode
        if quiet is False:
            if rgi is None:
                logger.warning("RGI object is required for QC plots but was not provided.")
            else:
                self._generate_qc_plots(rgiid, name, data_dir, x, zb, h0, df, rgi)

        # Store attributes
        self.x = df.x
        self.zb = df.zb
        self.w = df.w
        self.h0 = df.h0
    
    
    def _generate_qc_plots(self, rgiid, name, data_dir, x, zb, h0, df, rgi):
        """
        Generate quality control plots for the flowline geometry.
        
        Parameters
        ----------
        rgiid : str
            RGI ID of the glacier.
        name : str
            Name of the glacier.
        data_dir : Path
            Directory to save plots.
        x : numpy.ndarray
            Horizontal distance along flowline.
        zb : numpy.ndarray
            Original bed elevation.
        h0 : numpy.ndarray
            Original ice thickness.
        df : pandas.DataFrame
            Processed flowline data.
        rgi : object
            RGI glacier object containing metadata like area_km2 and lmax_m.
        """
        # Plot 1: Bed elevation and ice thickness
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=200)
        ax.plot(x, zb, color='black', label='Original zb')
        ax.plot(df['x'], df['zb'], color='red', label='New zb')
        ax.plot(df['x'], df['zb'] + df['h0'], color='blue', label='New h0')
        ax.plot(x, zb + h0, color='black', ls='--', label='Original h0')
        ax.grid()
        ax.legend()
        ax.set_title(f'{rgiid} {name}')
        ax.set_axisbelow(True)
        plot1_path = data_dir / f'flowline_geom_QC_{rgiid}_{name.lower().replace(" ", "")}.png'
        plt.savefig(plot1_path)
        fig.show()

        # Plot 2: Ice surface and bed profile
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=200)
        idx_term = fl.terminus_index
        idx = idx_term + 60
        ax.plot(df.x[:idx], df.zb[:idx] + df.h0[:idx], c='blue', label='Ice surface')
        ax.plot(df.x[:idx], df.zb[:idx], c='black', label='Bed elevation')
        ax.axvline(rgi.lmax_m, c='red', label=f'RGI Length ({rgi.lmax_m} m)')
        ax.set_title(f'{rgiid} {name}')
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()
        plot2_path = data_dir / f'flowline_geom_profile_{rgiid}_{name.lower().replace(" ", "")}.png'
        plt.savefig(plot2_path)
        fig.show()

        # Plot 3: Width and area
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=200)
        ax.plot(df.x[:idx], df.w[:idx] / 1000, c='black', label='w')
        ax.plot(df.x[:idx_term], np.cumsum(df.w[:idx_term] * 50) / 1e6, c='grey', 
                label=f'Flowline area: {df.w[:idx_term].sum() * 50/1e6:.2f}')
        ax.axhline(rgi.area_km2, c='red', ls='--', label=f'RGI area_km2 {rgi.area_km2:.2f}')
        ax.axvline(rgi.lmax_m, c='red', label=f'RGI Length ({rgi.lmax_m} m)')
        ax.set_title(f'{rgiid} {name}')
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend()
        plot3_path = data_dir / f'flowline_geom_area_{rgiid}_{name.lower().replace(" ", "")}.png'
        plt.savefig(plot3_path)
        fig.show()


    def to_pandas(self) -> pd.DataFrame:
        """
        Convert flowline geometry data to a pandas DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing x, zb, w, and h0 columns representing horizontal distance,
            bed elevation, width, and ice thickness respectively.
        """
        return pd.DataFrame(dict(x=self.x, zb=self.zb, w=self.w, h0=self.h0))



def extract_flowline_geom(config_path: Optional[str] = None):
    """
    Extract flowline geometry from OGGM workflow.
    
    This function sets up and runs the OGGM workflow to extract flowline geometries
    for a set of glaciers. It performs several steps:
    1. Initialize OGGM configuration
    2. Set up glacier directories
    3. Execute OGGM tasks (DEM processing, flowline extraction, climate data, inversion)
    4. Save processed data to output directory
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. If None, uses default configuration.
        
    Notes
    -----
    This function requires OGGM to be properly installed and configured.
    Output files are saved to the directory specified in the configuration.
    """
    import oggm
    from oggm import workflow
    from oggm import tasks
    from config import cfg

    # Helper function to flatten nested lists
    def flatten(items, seqtypes=(list, tuple)):
        try:
            for i, x in enumerate(items):
                while isinstance(x, seqtypes):
                    items[i:i + 1] = x
                    x = items[i]
        except IndexError:
            pass
        return items

    # Initialize OGGM
    oggm.cfg.initialize(logging_level='DEBUG')
    
    # Set OGGM paths - these should be configurable from environment or config file
    oggm.cfg.PATHS['dl_cache_dir'] = Path(os.environ.get('OGGM_DL_CACHE', r'~/OGGM/download_cache'))
    oggm.cfg.PATHS['rgi_dir'] = Path(os.environ.get('OGGM_RGI_DIR', r'~/OGGM/rgi'))
    oggm.cfg.PATHS['test_dir'] = Path(os.environ.get('OGGM_TEST_DIR', r'~/OGGM/tests'))
    oggm.cfg.PATHS['tmp_dir'] = Path(os.environ.get('OGGM_TMP_DIR', r'~/OGGM/tmp'))
    oggm.cfg.PATHS['working_dir'] = Path(os.environ.get('OGGM_WORKING_DIR', r'~/oggm_out'))
    
    # Set OGGM parameters
    oggm.cfg.PARAMS['prcp_scaling_factor'] = 2.5
    oggm.cfg.PARAMS['climate_qc_months'] = 3
    oggm.cfg.PARAMS['use_winter_prcp_factor'] = False
    oggm.cfg.PARAMS['min_mu_star'] = 50
    oggm.cfg.PARAMS['max_mu_star'] = 1000
    oggm.cfg.PARAMS['use_multiprocessing'] = False
    oggm.cfg.PARAMS['check_calib_params'] = False
    oggm.cfg.PARAMS['use_rgi_area'] = True
    oggm.cfg.PARAMS['use_tstar_calibration'] = True
    oggm.cfg.PARAMS['flowline_dx'] = 1.0
    oggm.cfg.PARAMS['elevation_band_flowline_binsize'] = 30  # meters
    oggm.cfg.PARAMS['grid_dx_method'] = 'fixed'
    oggm.cfg.PARAMS['fixed_dx'] = 50
    oggm.cfg.PARAMS['border'] = 240
    oggm.cfg.PARAMS['use_multiple_flowlines'] = False
    oggm.cfg.PARAMS['downstream_line_shape'] = 'parabola'
    
    # Get RGI IDs from config
    try:
        rgi_ids = list({k: v for k, v in cfg['glaciers'].items()}.keys())
    except KeyError:
        logger.error("Configuration missing 'glaciers' key")
        raise

    # Set output directory
    output_dir = os.environ.get('OGGM_OUTPUT_DIR', '~/oggm_out/e_bands')
    oggm.cfg.PATHS['working_dir'] = Path(output_dir)
    
    # Define pre-processed directory URLs
    # todo: just until 1.6 transition is documented then this should be changed
    prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands'
    base_url = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v1/highres/'
    
    # Initialize glacier directories
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=2, prepro_border=160,
                                            prepro_base_url=prepro_path, reset=False, force=False)

    # Execute OGGM tasks
    # 1. Shared tasks
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                               source=['ALASKA', 'NASADEM'])
    workflow.execute_entity_task(tasks.process_dem, gdirs)

    # 2. Get the flowline width using elevation band flowlines
    workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs, write_hypsometry=True)
    workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs)
    workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs)
    workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
    workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
    
    # 3. Climate tasks
    base_url = r'https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/RGIV62/CRU/elev_bands/qc3/pcp2.5/'
    workflow.download_ref_tstars(base_url=base_url)
    list_tasks = [
        tasks.process_climate_data,
        tasks.local_t_star,
        tasks.mu_star_calibration
    ]
    for task in list_tasks:
        workflow.execute_entity_task(task, gdirs)
    
    # 4. Inversion tasks
    workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs, invert_all_rectangular=True)
    list_tasks = [
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output,
        tasks.gridded_attributes,
    ]
    for task in list_tasks:
        workflow.execute_entity_task(task, gdirs)

    # 5. Init model
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # 6. Copy files to more convenient location (outside OGGM environment)
    # Get output directory from environment or config
    external_output_dir = os.environ.get(
        'FLOWLINE_OUTPUT_DIR', 
        '/mnt/c/sandbox/glacier-attribution/data/interim/oggm_flowlines/e_bands'
    )
    external_output_path = Path(external_output_dir)
    external_output_path.mkdir(parents=True, exist_ok=True)
    
    for i, rgiid in enumerate(rgi_ids):
        gdir = gdirs[i]
        gdir_objs = ['inversion_flowlines', 'inversion_output', 'downstream_line', 'elevation_band_flowline']
        
        for gdir_obj in gdir_objs:
            fp = gdir.get_filepath(gdir_obj)
            
            if gdir_obj == 'elevation_band_flowline':
                output_file = external_output_path / f'{rgiid}.{gdir_obj}.csv'
                shutil.copyfile(fp, output_file)
            else:
                # Read the object
                objs = []
                try:
                    with gzip.open(fp, "rb") as openfile:
                        while True:
                            try:
                                obj = pickle.load(openfile)
                                objs.append(obj)
                            except EOFError:
                                break
                    
                    # Write to external location
                    output_file = external_output_path / f'{rgiid}.{gdir_obj}.pickle'
                    with gzip.open(output_file, "wb") as f:
                        # Flatten the objects and remove dependencies on oggm/salem
                        objs = flatten(objs)
                        pickle.dump(objs, f)
                except Exception as e:
                    logger.error(f"Error processing {gdir_obj} for {rgiid}: {e}")
                    
                    
                    
if __name__ == "__main__":
    # Example usage
    # 1. Extract flowline geometry data
    # extract_flowline_geom()
    
    # 2. Process a specific glacier
    geom = get_flowline_geom(rgiid="RGI60-11.00897", name="Hintereisferner")
    
    # 3. Access the geometry data
    df = geom.to_pandas()
    print(f"Flowline length: {df['x'].max()} m")
    print(f"Elevation range: {df['zb'].min()} - {df['zb'].max()} m")