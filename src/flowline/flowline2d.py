# -*- coding: utf-8 -*-
"""
flowline2d.py

Description.

Author: drotto
Created: 5/2/2023 @ 10:38 AM
Project: glacier-attribution
"""

import copy
import collections
import traceback
from functools import partial
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import dill
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy as sci
import scipy.io
import scipy.ndimage
import xarray as xr
import numba as nb
from numpy.random import default_rng
from scipy.interpolate import interp1d
from scipy.stats import norm
import logging
from tqdm import tqdm
# import gm  # Commented out - local package not currently installed


# Custom exceptions
class FlowlineModelError(Exception):
    """Base exception for flowline model errors"""
    pass


class GeometryError(FlowlineModelError):
    """Errors related to geometry setup"""
    pass


class NumericalInstabilityError(FlowlineModelError):
    """Errors from numerical instabilities"""
    pass


@dataclass
class FlowlineConfig:
    """Configuration parameters for the flowline model"""
    # Physical parameters
    rho: float = 916.8  # Ice density kg/m^3
    g: float = 9.81     # Gravity m/s^2
    fd: float = 1.9e-24 # Deformation parameter Pa^-3 s^-2
    fs: float = 5.7e-20 # Sliding parameter Pa^-3 s^-1 m^2
    n: int = 3          # Glenn's flow law parameter
    k: int = 3          # Sliding law parameter
    
    # Numerical parameters
    delx: float = 50           # Grid spacing in m
    delt: float = 0.0125 / 8   # Time step in yrs
    ts: float = 0              # Starting time yr
    tf: float = 2025           # Ending time yr
    min_thick: float = 1       # Minimum thickness for ice at terminus
    
    # Output parameters
    deltout: float = 1         # Frequency to save output
    dt_plot: int = 100         # Plotting interval yr
    rt_plot: bool = False      # Real time plotting
    xlim0: float = None        # Left limit for plots
    
    # Climate parameters
    gamma: float = 6.5e-3      # Temperature lapse rate degC/km
    mu: float = 0.65           # Melt rate m/yr/degC
    hmb: bool = False           # Height mass balance feedback
    
    def __post_init__(self):
        # Convert deformation parameters from seconds to years
        self.fd = self.fd * np.pi * 1e7
        self.fs = self.fs * np.pi * 1e7
        
        # Validation
        if self.tf <= self.ts:
            raise ValueError("tf must be greater than ts")
        if self.delx <= 0:
            raise ValueError("delx must be positive")
        if self.delt <= 0:
            raise ValueError("delt must be positive")


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
        elif self.x_init is not None and self.h_init is not None:
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
            raise GeometryError(f"Error during initial profile interpolation: {e}")

        return profile_source


class MassBalanceForcing(ABC):
    """Base class for mass balance forcing"""
    
    @abstractmethod
    def get_mass_balance(self, x, h_eff, year_idx):
        """Calculate mass balance for given conditions"""
        pass
    
    @abstractmethod
    def get_climate_vars(self, year_idx):
        """Get climate variables for output"""
        pass


class TemperaturePrecipitationForcing(MassBalanceForcing):
    """Temperature-precipitation based mass balance forcing"""
    
    def __init__(self, T0, P0, sigT=1, sigP=1, T=None, P=None, temp=None, 
                 t_stab=None, mu=0.65, gamma=6.5e-3, dpdz=None, T2melt=None,
                 pdd_Tamp=None, pdd_beta=None, ts=0, tf=2025):
        self.T0 = T0
        self.P0 = P0
        self.sigT = sigT
        self.sigP = sigP
        self.mu = mu
        self.gamma = gamma
        self.T2melt = T2melt
        self.pdd_Tamp = pdd_Tamp
        self.pdd_beta = pdd_beta
        self.ts = ts
        
        nyrs = int(np.ceil(tf - ts))
        
        # Initialize climate arrays
        if T is None:
            T = np.zeros(nyrs)
        if P is None:
            P = np.zeros(nyrs)
        if temp is None:
            temp = np.zeros(nyrs)
        if dpdz is None:
            dpdz = np.zeros(5000)  # Default elevation range
            
        self.Tp = sigT * T  # Temperature perturbation
        self.Pp = sigP * P  # Precipitation perturbation
        self.temp = temp    # Temperature trend
        self.dpdz = dpdz    # Precipitation-elevation relationship
        
        # Apply stability period
        if t_stab:
            self.Tp[:t_stab] = 0
            self.Pp[:t_stab] = 0
            self.temp[:t_stab] = 0
    
    def get_mass_balance(self, x, h_eff, year_idx):
        """Calculate mass balance from temperature and precipitation"""
        accumulation = (self.P0 + self.Pp[year_idx]) * np.ones(x.size)
        T_wk = ((self.T0 + self.Tp[year_idx]) * np.ones(x.size) + 
                self.temp[year_idx] - self.gamma * h_eff)
        
        pdd = None
        if callable(self.T2melt):
            melt = self.T2melt(T_wk)
        elif self.T2melt == 'pdd':
            pdd = calc_pdd(T_wk, self.pdd_Tamp)
            melt = np.maximum(0, pdd * self.mu)
        else:
            melt = np.maximum(0, T_wk * self.mu)
        
        return accumulation - melt, {'accumulation': accumulation, 'melt': melt, 'T': T_wk, 'pdd': pdd}
    
    def get_climate_vars(self, year_idx):
        """Get climate variables for output"""
        return {
            'T': self.T0 + self.Tp[year_idx] + self.temp[year_idx]
        }


class DirectMassBalanceForcing(MassBalanceForcing):
    """Direct mass balance forcing with optional spatial gradients and temporal anomalies"""
    
    def __init__(self, b0=0, bp=None, dbdz=None, dbdx=None):
        """
        Initialize direct mass balance forcing
        
        Parameters
        ----------
        b0 : float
            Base mass balance rate (m/yr)
        bp : float, array-like, or None
            Mass balance anomaly time series (m/yr). Can be:
            - float: constant anomaly for all time
            - array: time series of anomalies (one value per year)
            - None: no anomaly (default)
        dbdz : array-like or None
            Mass balance gradient with elevation (m/yr per m elevation).
            Array should be indexed by elevation in meters.
        dbdx : array-like or None  
            Mass balance gradient with distance (m/yr per m distance).
            Array should be indexed by distance in meters.
        """
        self.b0 = b0
        self.dbdz = dbdz
        self.dbdx = dbdx
        
        # Handle bp (mass balance anomaly)
        if bp is None:
            self.bp = None
        elif np.isscalar(bp):
            self.bp = bp  # Constant anomaly
        else:
            self.bp = np.array(bp)  # Time series
    
    def get_mass_balance(self, x, h_eff, year_idx):
        """Calculate mass balance directly"""
        # Start with base mass balance
        b = np.full_like(x, self.b0, dtype=float)
        
        # Add elevation-dependent component
        if self.dbdz is not None:
            # Clip elevation indices to valid range to prevent crashes
            h_indices = np.clip(h_eff.astype(int), 0, len(self.dbdz) - 1)
            b += self.dbdz[h_indices]
        
        # Add distance-dependent component  
        if self.dbdx is not None:
            # Clip distance indices to valid range to prevent crashes
            x_indices = np.clip(x.astype(int), 0, len(self.dbdx) - 1)
            b += self.dbdx[x_indices]
        
        # Add temporal anomaly
        bp_val = 0.0
        if self.bp is not None:
            if np.isscalar(self.bp):
                # Constant anomaly
                bp_val = self.bp
            else:
                # Time series anomaly
                bp_val = self.bp[year_idx]
            b += bp_val

        accumulation = np.maximum(0, b)
        melt = np.maximum(0, -b)
        
        return b, {'b_anomaly': bp_val, 'accumulation': accumulation, 'melt': melt}
    
    def get_climate_vars(self, year_idx):
        """Get climate variables for output"""
        return {}


class flowline2d:
    def __init__(self, config=None, geometry=None, forcing=None, **kwargs):
        """2d flowline model with modular configuration
        
        Parameters
        ----------
        config : FlowlineConfig, optional
            Model configuration parameters
        geometry : FlowlineGeometry, optional  
            Glacier geometry setup
        forcing : MassBalanceForcing, optional
            Mass balance forcing method
        **kwargs : dict
            Additional parameters for backward compatibility
        """
        
        # Handle backward compatibility
        if config is None or geometry is None or forcing is None:
            return self._init_legacy(**kwargs)
        
        self.config = config
        self.geometry = geometry
        self.forcing = forcing
        self.no_error = True
        
        # Setup model
        self._setup_model()

    def _init_legacy(self, **kwargs):
        """Legacy initialization for backward compatibility"""
        # Extract required parameters
        x_gr = kwargs.pop('x_gr')
        zb_gr = kwargs.pop('zb_gr') 
        w_geom = kwargs.pop('w_geom')
        mode = kwargs.pop('mode', 'TP')
        
        # Create config from kwargs
        config_params = {}
        for key in ['rho', 'g', 'fd', 'fs', 'n', 'k', 'delx', 'delt', 'ts', 'tf', 
                   'min_thick', 'deltout', 'dt_plot', 'rt_plot', 'xlim0', 'gamma', 'mu', 'hmb']:
            if key in kwargs:
                config_params[key] = kwargs.pop(key)
        
        self.config = FlowlineConfig(**config_params)
        
        # Create geometry
        self.geometry = FlowlineGeometry(
            x_gr, zb_gr, w_geom,
            x_init=kwargs.get('x_init') or kwargs.get('x_geom'),
            h_init=kwargs.get('h_init') or kwargs.get('h_geom'),
            profile=kwargs.get('profile')
        )
        
        # Create forcing based on mode
        if mode == 'TP':
            self.forcing = TemperaturePrecipitationForcing(
                T0=kwargs.get('T0'), P0=kwargs.get('P0'),
                sigT=kwargs.get('sigT', 1), sigP=kwargs.get('sigP', 1),
                T=kwargs.get('T'), P=kwargs.get('P'), temp=kwargs.get('temp'),
                t_stab=kwargs.get('t_stab'), mu=self.config.mu, gamma=self.config.gamma,
                dpdz=kwargs.get('dpdz'), T2melt=kwargs.get('T2melt'),
                pdd_Tamp=kwargs.get('pdd_Tamp'), pdd_beta=kwargs.get('pdd_beta'),
                ts=self.config.ts, tf=self.config.tf
            )
        elif mode == 'b':
            # Combine bp and bal for backward compatibility
            bp_combined = kwargs.get('bp')
            bal = kwargs.get('bal')
            
            if bp_combined is not None and bal is not None:
                # If both exist, add them together
                if np.isscalar(bp_combined) and np.isscalar(bal):
                    bp_final = bp_combined + bal
                else:
                    bp_combined = np.atleast_1d(bp_combined)
                    bal = np.atleast_1d(bal)
                    # Ensure same length, pad with zeros if needed
                    max_len = max(len(bp_combined), len(bal))
                    bp_padded = np.pad(bp_combined, (0, max_len - len(bp_combined)), 'constant')
                    bal_padded = np.pad(bal, (0, max_len - len(bal)), 'constant')
                    bp_final = bp_padded + bal_padded
            elif bp_combined is not None:
                bp_final = bp_combined
            elif bal is not None:
                bp_final = bal
            else:
                bp_final = None
            
            self.forcing = DirectMassBalanceForcing(
                b0=kwargs.get('b0', 0),
                bp=bp_final,
                dbdz=kwargs.get('bz'),  # Rename bz to dbdz for clarity
                dbdx=kwargs.get('bx')   # Rename bx to dbdx for clarity
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.no_error = True
        self._setup_model()
    
    def _setup_model(self):
        """Setup model grid, geometry, and output arrays"""
        # Setup geometry and grid
        self.geometry.setup_grid(self.config.delx)
        self.spinup_result = self.geometry.load_initial_profile()
        
        # Copy geometry attributes for easy access
        self.x = self.geometry.x
        self.zb = self.geometry.zb
        self.w = self.geometry.w
        self.dzbdx = self.geometry.dzbdx
        self.dwdx = self.geometry.dwdx
        self.nxs = self.geometry.nxs
        self.h0 = self.geometry.h0
        
        # Store original geometry grid for posterity
        if self.spinup_result:
            self.x_gr = self.spinup_result.geometry.x_gr
            self.zb_gr = self.spinup_result.geometry.zb_gr
            self.w_geom = self.spinup_result.geometry.w_geom
        else:
            self.x_gr = self.geometry.x_gr
            self.zb_gr = self.geometry.zb_gr
            self.w_geom = self.geometry.w_geom

        # Calculate number of time steps
        self.nts = round(np.floor((self.config.tf - self.config.ts) / self.config.delt))
        
        # Initialize output arrays
        self._initialize_output_arrays()
    
    def _initialize_output_arrays(self):
        """Initialize all output arrays"""
        nouts = int((self.nts * self.config.delt) // self.config.deltout)
        
        # Common outputs
        self.edge_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.edge = np.full(nouts, fill_value=np.nan, dtype="float")
        self.t = np.full(nouts, fill_value=np.nan, dtype="float")
        self.total_mass_balance = np.full(nouts, fill_value=np.nan, dtype="float")
        self.ela = np.full(nouts, fill_value=np.nan, dtype="float")
        self.area = np.full(nouts, fill_value=np.nan, dtype="float")
        self.h = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.b_profile = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.b_anomaly = np.full(nouts, fill_value=np.nan, dtype="float")
        self.accumulation = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.melt = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.ela_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.F = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        
        # Climate-specific outputs
        if isinstance(self.forcing, TemperaturePrecipitationForcing):
            self.T = np.full(nouts, fill_value=np.nan, dtype="float")
            if hasattr(self.forcing, 'T2melt') and self.forcing.T2melt == 'pdd':
                self.pdd = np.full((nouts, self.nxs), fill_value=0.0, dtype="float")

    def run(self, **kwargs):
        """Single entry point for running the model"""
        # Update config with any runtime overrides
        if kwargs:
            config_dict = asdict(self.config)
            config_dict.update(kwargs)
            self.config = FlowlineConfig(**config_dict)
            # Recalculate dependent values
            self.nts = round(np.floor((self.config.tf - self.config.ts) / self.config.delt))
            self._initialize_output_arrays()
        
        try:
            return self._run_model()
        except Exception as e:
            self.no_error = False
            raise FlowlineModelError(f"Model run failed: {e}") from e

    def _run_model(self):
        """Unified model run method"""
        yr = self.config.ts - 1  # -1 because we increment at start of loop
        idx_out = 0
        t_out = 0.0  # Next time to save output
        b = np.zeros(self.nxs)
        climate_vars = {}

        h = self.h0.copy()  # Initial thickness
        
        for i in tqdm(
            range(0, self.nts),
            unit_scale=self.config.delt,
            unit="yrs",
            bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
            ascii=True,
            ncols=100,
        ):
            t = self.config.delt * i  # time in fractional years

            # Update climate on integer year change
            current_model_year = self.config.ts + int(np.floor(t))
            if current_model_year > yr:
                yr = current_model_year
                year_idx = yr - self.config.ts

                # Calculate effective height for mass balance
                if self.config.hmb:
                    h_eff = self.zb + h
                else:
                    h_eff = self.zb

                # Get mass balance from forcing
                b, climate_vars = self.forcing.get_mass_balance(
                    self.x, h_eff, year_idx
                )

            # Solve shallow ice approximation
            h, edge_idx, F = space_loop(
                h, b, self.x, self.config.rho, self.config.g, self.nxs,
                self.config.delx, self.dzbdx, self.config.fd, self.config.fs,
                self.dwdx, self.w, self.config.delt, self.config.min_thick,
                self.config.n, self.config.k
            )

            # Check for numerical instability
            if np.any(np.isnan(h)):
                self.no_error = False
                error_msg = (
                    f"Numerical instability detected at model time t = {t:.2f} years.\n"
                    f"NaN values found in thickness 'h'.\n"
                    f"Diagnostics (at last stable step):\n"
                    f"  - Max h: {np.nanmax(h) if not np.all(np.isnan(h)) else 'all NaN'}\n"
                    f"  - Max b: {np.nanmax(b)}\n"
                    f"  - Max F: {np.nanmax(F)}\n"
                )
                raise NumericalInstabilityError(error_msg)

            # Save output at specified interval
            if t >= t_out and idx_out < len(self.t):
                self._save_output(idx_out, t, h, b, edge_idx, F, climate_vars)
                idx_out += 1
                t_out += self.config.deltout

        # Check for successful completion
        self.no_error = not np.isnan(self.h[-1, 0])
        return copy.deepcopy(self)
    
    def _save_output(self, idx_out, t, h, b, edge_idx, F, climate_vars):
        """Save model output at current time step"""
        area = np.sum(self.w[:edge_idx]) * self.config.delx
        mass_balance_flux = b * self.w * self.config.delx
        
        # Common outputs
        self.t[idx_out] = t + self.config.ts
        self.edge_idx[idx_out] = edge_idx
        self.edge[idx_out] = edge_idx * self.config.delx
        self.h[idx_out, :] = h
        self.area[idx_out] = area
        self.total_mass_balance[idx_out] = mass_balance_flux[:edge_idx].sum()
        self.b_profile[idx_out, :] = b
        self.F[idx_out, :] = F
        
        # Calculate ELA
        try:
            ela_idx = np.abs(b[:edge_idx]).argmin()
        except:
            ela_idx = 0
        self.ela_idx[idx_out] = ela_idx
        self.ela[idx_out] = self.zb[ela_idx] + h[ela_idx]
        
        # Save detailed mass balance components from climate_vars
        self.b_anomaly[idx_out] = climate_vars.get('b_anomaly', np.nan)
        if 'accumulation' in climate_vars:
            self.accumulation[idx_out, :] = climate_vars['accumulation']
        if 'melt' in climate_vars:
            self.melt[idx_out, :] = climate_vars['melt']
        
        # Climate-specific outputs
        if isinstance(self.forcing, TemperaturePrecipitationForcing):
            climate_out = self.forcing.get_climate_vars(
                int((t + self.config.ts - self.config.ts))
            )
            if 'T' in climate_out:
                self.T[idx_out] = climate_out['T']
            if hasattr(self, 'pdd') and 'pdd' in climate_vars and climate_vars['pdd'] is not None:
                # Ensure PDD values are non-negative and handle NaN
                pdd_values = climate_vars['pdd']
                pdd_values = np.where(np.isnan(pdd_values), 0.0, pdd_values)
                pdd_values = np.maximum(pdd_values, 0.0)
                self.pdd[idx_out, :] = pdd_values


    def to_pickle(self, fp):
        with open(fp, "wb") as f:
            dill.dump(self, f)

        return None

    def to_pandas(self):
        d = dict(
            area=self.area,
            total_mass_balance=self.total_mass_balance,
            edge=self.edge_idx,
            edge_m=self.edge,
            ela=self.ela,
        )
        
        # Only add T if it exists (for temperature-precipitation forcing)
        if hasattr(self, 'T') and self.T is not None:
            d['T'] = self.T
            
        df = pd.DataFrame(d, index=self.t)
        return df

    def to_xarray(self):
        """Convert results to xarray Dataset with proper metadata"""
        # Build data variables dynamically based on what exists
        data_vars = {
            'edge_idx': (['time'], self.edge_idx),
            'edge': (['time'], self.edge),
            'total_mass_balance': (['time'], self.total_mass_balance),
            'b_profile': (['time', 'x'], self.b_profile),
            'b_anomaly': (['time'], self.b_anomaly),
            'accumulation': (['time', 'x'], self.accumulation),
            'melt': (['time', 'x'], self.melt),
            'ela': (['time'], self.ela),
            'h': (['time', 'x'], self.h),
            'area': (['time'], self.area),
            'w': (['x'], self.w),
            'zb': (['x'], self.zb),
            'F': (['time', 'x'], self.F),
        }
        
        # Add climate-specific variables if they exist
        if hasattr(self, 'T') and self.T is not None:
            data_vars['T'] = (['time'], self.T)
        if hasattr(self, 'pdd') and self.pdd is not None:
            data_vars['pdd'] = (['time', 'x'], self.pdd)
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': self.t,
                'x': self.x,
            },
            attrs=asdict(self.config)
        )
        
        # Add spin-up metadata if available
        if hasattr(self, 'spinup_result') and self.spinup_result is not None:
            if isinstance(self.spinup_result, flowline2d):
                spinup_config = asdict(self.spinup_result.config)
                for k, v in spinup_config.items():
                    # Avoid overwriting main config attributes
                    if f'spinup_{k}' not in ds.attrs:
                         ds.attrs[f'spinup_{k}'] = str(v) # Convert to string for safety
            elif isinstance(self.spinup_result, str):
                ds.attrs['spinup_profile_path'] = self.spinup_result

        return ds

    def copy(self):
        return copy.deepcopy(self)

    def calc_diag(res, t=(None, None)):
        tslice = slice(t[0], t[1])

        diag = pd.DataFrame(dtype=float, columns=['mean', 'std', 'mean_025', 'mean_975', 'std_025', 'std_975'])
        df = len(res.edge)
        b = res.total_mass_balance / res.area
        diag.loc['b', 'mean'] = b[tslice].mean()
        diag.loc['b', 'std'] = b[tslice].std()
        diag.loc['b', 'mean_025'], diag.loc['b', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['b', 'mean'], scale=diag.loc['b', 'std']
        )
        # diag.loc['b', 'std_025'], diag.loc['b', 'std_975'] = gm.std_cinterval(b, 0.95)  # gm not available
        diag.loc['b', 'std_025'] = diag.loc['b', 'std_975'] = np.nan
        try:
            diag.loc['T', 'std'] = res.T[tslice].mean(axis=1).std()
            diag.loc['accumulation', 'std'] = res.accumulation[tslice].mean(axis=1).std()
        except:
            pass
        diag.loc['L', 'mean'] = res.edge[tslice].mean()
        diag.loc['L', 'std'] = res.edge[tslice].std()
        diag.loc['L', 'mean_025'], diag.loc['L', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['L', 'mean'], scale=diag.loc['L', 'std']
        )
        # diag.loc['L', 'std_025'], diag.loc['L', 'std_975'] = gm.std_cinterval(res.edge[tslice], 0.95)  # gm not available
        diag.loc['L', 'std_025'] = diag.loc['L', 'std_975'] = np.nan
        diag.loc['Hmax', 'mean'] = res.h[tslice].max(axis=1).mean()
        diag.loc['Hmax', 'std'] = res.h[tslice].max(axis=1).std()
        diag.loc['Hmax', 'mean_025'], diag.loc['Hmax', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['Hmax', 'mean'], scale=diag.loc['Hmax', 'std']
        )
        diag.loc['Area', 'mean'] = res.area[tslice].mean() / 1e6
        diag.loc['Area', 'std'] = res.area[tslice].std() / 1e6
        diag.loc['Area', 'mean_025'], diag.loc['Area', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['Area', 'mean'], scale=diag.loc['Area', 'std']
        )
        diag.loc['ELA', 'mean'] = res.ela[tslice].mean()
        diag.loc['ELA', 'std'] = res.ela[tslice].std()
        diag.loc['ELA', 'mean_025'], diag.loc['ELA', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['ELA', 'mean'], scale=diag.loc['ELA', 'std']
        )
        babl = np.array([res.b_profile[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
        bacc = np.array([res.b_profile[i, :j].mean() for i, j in enumerate(res.ela_idx[tslice])])
        diag.loc['babl', 'mean'] = np.nanmean(babl)
        diag.loc['bacc', 'mean'] = np.nanmean(bacc)
        diag.loc['babl', 'std'] = np.nanstd(babl)
        diag.loc['bacc', 'std'] = np.nanstd(bacc)
        diag.loc['babl', 'mean_025'], diag.loc['babl', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['babl', 'mean'], scale=diag.loc['babl', 'std']
        )
        diag.loc['bacc', 'mean_025'], diag.loc['bacc', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['bacc', 'mean'], scale=diag.loc['bacc', 'std']
        )
        Habl = np.array([res.h[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
        w = res.w.reshape(1, -1).repeat(10000, 0)
        wabl = np.array([w[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
        diag.loc['Habl', 'mean'] = Habl.mean()
        diag.loc['Habl', 'std'] = Habl.std()
        diag.loc['Habl', 'mean_025'], diag.loc['Habl', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['Habl', 'mean'], scale=diag.loc['Habl', 'std']
        )
        diag.loc['wabl', 'mean'] = wabl.mean()
        diag.loc['wabl', 'std'] = wabl.std()
        beta = res.area[tslice] / (Habl * wabl)
        diag.loc['beta', 'mean'] = beta.mean()
        diag.loc['beta', 'std'] = beta.std()
        aar = np.array([w[i, 0:j].sum() * res.delx for i, j in enumerate(res.ela_idx[tslice])]) / res.area[tslice]
        diag.loc['aar', 'mean'] = aar.mean()
        diag.loc['aar', 'std'] = aar.std()
        return diag

    @property
    def beta(self):
        w = self.w.reshape(1, -1).repeat(10000, 0)
        Habl = np.array([self.h[i, j[0] : j[1]].mean() for i, j in enumerate(zip(self.ela_idx, self.edge_idx))])
        wabl = np.array([w[i, j[0] : j[1]].mean() for i, j in enumerate(zip(self.ela_idx, self.edge_idx))])
        beta = self.area / (Habl * wabl)
        return beta

    def calc_tau(self):
        H = np.array([self.h[i, (self.ela_idx[i]) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        bt = np.array([self.b_profile[i, (self.ela_idx[i] - 10) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        tau = -H / bt
        return tau, H, bt

    def calc_tau2(self, t_idx):
        return (
            -self.h[np.arange(self.h.shape[0]), self.edge_idx - t_idx]
            / self.b_profile[np.arange(self.b_profile.shape[0]), self.edge_idx - t_idx]
        )

    def calc_tau4(self, mu=None, gamma=None):
        w = self.w.reshape(1, -1).repeat(10000, 0)
        w = np.array([w[i, (self.ela_idx[i]) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        H = np.array([self.h[i, (self.ela_idx[i]) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        Aabl = (self.edge_idx - self.ela_idx) * self.delx * w
        zb = self.zb.reshape(1, -1).repeat(10000, 0)
        tanphi = np.gradient(zb, axis=1) / self.delx
        tanphi = np.array([tanphi[i, (self.ela_idx[i]) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        if self.mu is not None:
            mu = self.mu
        if self.gamma is not None:
            gamma = self.gamma
        tau = -(w * H) / (mu * gamma * tanphi * Aabl)
        return tau.mean()

    def calc_tau_from_acf(self):
        def fit_acf(t, tau):
            eps = 1 / np.sqrt(3)
            acf = np.exp(-t / (eps * tau)) * (1 + t / (eps * tau) + 1 / 3 * (t / (eps * tau)) ** 2)
            return acf

        res = self.copy()
        t = 200
        # acx = gm.acf(res.edge, t)  # gm not available
        acx = np.correlate(res.edge, res.edge, mode='full')  # Simple placeholder
        out = sci.optimize.curve_fit(
            fit_acf,
            np.arange(0, t),
            acx,
        )
        tau = out[0][0]
        return tau

    def calc_tau_from_psd(self):
        def calc_psd(L):
            M_window = 1024
            n_overlap = M_window // 2
            f, Pxx = sci.signal.welch(L, nperseg=1024, noverlap=512, detrend='linear')
            return f, Pxx

        def fit_psd(f, tau, sigb, beta):
            eps = 1 / np.sqrt(3)
            K = 1 - 1 / (eps * tau)
            P0 = beta**2 * tau**2 * sigb**2  # roe and baker 2016 eq. 7
            Pf = P0 * (1 - K) ** 6 / (1 - 2 * K * np.cos(2 * np.pi * f) + K**2) ** 3
            return Pf

        res = self.copy()
        f, Pyy = calc_psd(res.edge)
        calc_fitted_psd = partial(fit_psd, sigb=res.sigb, beta=res.beta.mean())
        out = sci.optimize.curve_fit(
            calc_fitted_psd,
            f,
            Pyy,
        )
        tau = out[0][0]
        return tau

    @property
    def Leq(self):
        """Equilibrium length"""
        return self.b_profile.mean(axis=1) * self.edge[0] / self.b_profile[np.arange(self.b_profile.shape[0]), self.edge_idx - 30]

    def calc_return(self, L0=0):
        Leq = self.Leq
        self.dLeq = Leq - self.edge
        excursions = self.dLeq > L0  # return time
        R = np.diff(np.where(np.concatenate(([excursions[0]], excursions[:-1] != excursions[1:], [True])))[0])[
            ::2
        ].mean()
        return R

    def calc_tau_from_return(self, R=None, L0=0):
        if R is None:
            R = self.calc_return(L0)
        sigdLeq = self.dLeq.std()
        return R / (2 * np.pi * np.exp(0.5 * L0 / sigdLeq))

    @property
    def tau_from_dLdt(self):
        """Response time from dL/dt"""
        sigdL = np.gradient(self.edge).std()
        sigL = self.edge.std()
        return sigL / sigdL


@nb.njit(fastmath={"contract", "arcp", "nsz", "afn", "reassoc"})
def space_loop(h, b, x, rho, g, nxs, delx, dzbdx, fd, fs, dwdx, w, delt, min_thick, n,k):
    Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
    Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
    dhdt = np.zeros(x.size)  # zero out thickness rate of change array
    rho_g_cu = (rho * g) ** n
    dzdx = (dzbdx[:-1] + dzbdx[1:]) / 2  # slope at plus half a grid point
    # -----------------------------------------
    # begin loop over space
    # -----------------------------------------
    for j in range(0, nxs - 1):
        if j == 0:
            h_ave = (h[0] + h[1]) / 2
            dhdx = (h[1] - h[0]) / delx
            Qp[0] = (
                -rho_g_cu * (dhdx + dzdx[j]) ** 3 * (fd * h_ave**(n+2) + fs * h_ave**k)  # top of glacier qp
            )  # flux at plus half grid point
            # Qm[0] = 0  # flux at minus half grid point
            dhdt[0] = b[0] - Qp[0] / (delx / 2) - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
        elif (h[j] <= min_thick) & (h[j - 1] > min_thick):  # glacier toe condition
            # Qp[j] = 0
            h_ave = h[j - 1] / 2
            dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
            Qm[j] = (
                -rho_g_cu * (dhdx + dzdx[j - 1]) ** 3 * (fd * h_ave**(n+2) + fs * h_ave**k)
            )  # glacier toe qm
            dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
        elif (h[j] <= min_thick) & (h[j - 1] <= min_thick):  # beyond glacier toe - no glacier flux
            dhdt[j] = b[j]
            # Qp[j] = 0
            # Qm[j] = 0
        else:  # within the glacier
            h_ave = (h[j + 1] + h[j]) / 2
            dhdx = (h[j + 1] - h[j]) / delx  # correction inserted ght nov-24-04
            Qp[j] = (
                -rho_g_cu * (dhdx + dzdx[j]) ** 3 * (fd * h_ave**(n+2) + fs * h_ave**k)
            )  # Within glacier qp
            h_ave = (h[j - 1] + h[j]) / 2
            dhdx = (h[j] - h[j - 1]) / delx
            Qm[j] = (
                -rho_g_cu * (dhdx + dzdx[j - 1]) ** 3 * (fd * h_ave**(n+2) + fs * h_ave**k)
            )  # within glacier qm
            dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
    # ----------------------------------------
    # end loop over space
    # ----------------------------------------
    #dhdt[nxs] = 0  # enforce no change at boundary
    h = np.core.umath.maximum(h + dhdt * delt, 0)
    
    # More robust edge detection
    edge = nxs - 1  # Default to end of domain
    for i in range(nxs):
        if h[i] < min_thick:
            edge = i
            break
    
    F = Qm + Qp
    return h, edge, F


def calc_ela(P0, T0, gamma, mu, h=None):
    # this seems to be accurate with elev mb feedback??
    if np.asarray(h).any():  # idk if this part is right
        T0 = T0 - h * gamma
    ela = T0 / gamma - P0 / (mu * gamma)
    return ela


def calc_Leq(A, w, bt, db, L=None):
    if np.ndim(w) != 0:
        w = np.mean(w)
    return A / w * -db / bt


#@nb.njit
def calc_tau3(h, b, edge_idx, toe_idx, term_idx):
    '''
    toe_idx = idx from terminus to start the terminus zone
    term_idx = idx after the start of the zone to end the zone
    '''
    n = h.shape[0]
    tau = np.empty(n)
    for i in range(n):
        j0, j1 = edge_idx[i] - toe_idx - term_idx, edge_idx[i] - toe_idx
        tau[i] = -h[i, j0:j1].mean() / b[i, j0:j1].mean()
    return tau


@nb.njit
def calc_pdd(T, Tamp, days=365):
    """
    Calculate Positive Degree Days from temperature data.
    
    Parameters:
    -----------
    T : numpy array
        Mean temperatures (°C)
    Tamp : float
        Temperature amplitude (daily variation, °C)
    days : int, optional
        Number of days in the period (default: 365)
        
    Returns:
    --------
    numpy array
        Positive degree days for each temperature point
    """
    # Create a copy to avoid modifying the input array
    pdd = np.zeros_like(T)
    
    for i in range(len(T)):
        if T[i] <= -Tamp:
            # If mean temp is very low, no positive temps will occur
            pdd[i] = 0
        else:
            # Semi-analytical solution for positive degree days
            # when temperature follows a sinusoidal pattern
            pdd[i] = 1/np.pi * (T[i] * np.arccos(-T[i]/Tamp) + 
                               Tamp * np.sqrt(1 - (T[i]/Tamp)**2))
    
    # Scale by the number of days in the period
    return pdd * days


def calc_b(z, P0, T0, gamma, mu, Tamp, days=365, return_pdd=False):
    """
    Calculate mass balance at elevation z.
    
    Parameters:
    -----------
    z : float or numpy array
        Elevation(s) in meters
    P0 : float
        Precipitation at reference level (m w.e.)
    T0 : float
        Temperature at reference level (°C)
    gamma : float
        Temperature lapse rate (°C/m)
    mu : float
        Melt factor (m w.e./°C-day)
    Tamp : float
        Temperature amplitude (°C)
    days : int, optional
        Number of days in the period (default: 365)
        
    Returns:
    --------
    float or numpy array
        Mass balance at elevation z (m w.e.)
    """
    # Precipitation (assumed constant with elevation for simplicity)
    P = P0
    
    # Temperature at elevation z
    T = T0 - (gamma * z)
    
    # Calculate melt from positive degree days
    pdd = calc_pdd(T, Tamp=Tamp, days=days)
    melt = pdd * mu 
    
    # Mass balance = accumulation - ablation
    mass_balance = P - melt

    # Return based on the return_pdd flag
    if return_pdd:
        return mass_balance, pdd
    else:
        return mass_balance
    

def fit_bprofile(b, bz, ba, z, P, T, gamma, mu, Tamp):
    out = sci.optimize.curve_fit(
        b,
        bz,
        ba,
        bounds=([P[0], T[0], gamma[0], mu[0], Tamp[0]], [P[1], T[1], gamma[1], mu[1], Tamp[1]]),
        sigma=np.full(len(ba), fill_value=0.1),
    )

    keys = ['P0', 'T0', 'gamma', 'mu', 'Tamp']
    bopt = {k: v for k, v in zip(keys, out[0])}

    z = np.arange(z[0], z[1])
    bopt_profile = b(z, *out[0])
    return bopt, bopt_profile
