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
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import matplotlib as mpl
mpl.use('Agg')
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

# Import from new geometry module
from .geometry import FlowlineGeometry
from .forcing import MassBalanceForcing, TemperaturePrecipitationForcing, DirectMassBalanceForcing
from .config import FlowlineConfig
from .utils import FlowlineModelError, NumericalInstabilityError


class flowline2d:
    def __init__(self, config: FlowlineConfig, geometry: FlowlineGeometry, forcing: MassBalanceForcing):
        """2d flowline model with modular configuration

        Parameters
        ----------
        config : FlowlineConfig
            Model configuration parameters
        geometry : FlowlineGeometry
            Glacier geometry setup
        forcing : MassBalanceForcing
            Mass balance forcing method
        """

        self.config = config
        self.geometry = geometry
        self.forcing = forcing
        self.no_error = True

        # Setup model
        self._setup_model()

    def _setup_model(self) -> None:
        """Setup model grid, geometry, and output arrays"""
        self.geometry.setup_grid(self.config.delx)

        # Copy geometry attributes for easy access
        self.x = self.geometry.x
        self.zb = self.geometry.zb
        self.w = self.geometry.w
        self.dzbdx = self.geometry.dzbdx
        self.dwdx = self.geometry.dwdx
        self.nxs = self.geometry.nxs
        self.h0 = self.geometry.h0
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
        # Extract no_progress parameter before updating config
        no_progress = kwargs.pop('no_progress', False)
        
        # Update config with any runtime overrides
        if kwargs:
            config_dict = asdict(self.config)
            config_dict.update(kwargs)
            self.config = FlowlineConfig(**config_dict)
            # Recalculate dependent values
            self.nts = round(np.floor((self.config.tf - self.config.ts) / self.config.delt))
            self._initialize_output_arrays()
        
        try:
            return self._run_model(no_progress=no_progress)
        except Exception as e:
            self.no_error = False
            raise FlowlineModelError(f"Model run failed: {e}") from e

    def _run_model(self, no_progress=False):
        """Unified model run method"""
        yr = self.config.ts - 1  # -1 because we increment at start of loop
        idx_out = 0
        t_out = 0.0  # Next time to save output
        b = np.zeros(self.nxs)
        climate_vars = {}

        h = self.h0.copy()  # Initial thickness
        
        if no_progress:
            range_iter = range(0, self.nts)
        else:
            range_iter = tqdm(
                range(0, self.nts),
                unit_scale=self.config.delt,
                unit="yrs",
                bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
                dynamic_ncols=True,   
            )
        
        for i in range_iter:
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
        
        # Target matching optimization hook
        if hasattr(self, 'target_matching') and self.target_matching:
            self._check_optimization_progress(idx_out, t, h, b, edge_idx)

    def _check_optimization_progress(self, idx_out, t, h, b, edge_idx):
        """
        Check optimization progress for target matching.
        
        This method is called from _save_output to monitor steady-state achievement
        and calculate cost functions for optimization.
        """
        # Update monitoring history
        current_time = t + self.config.ts
        self.time_history.append(current_time)
        
        # Calculate volume (simple approximation)
        volume = np.sum(h[:edge_idx] * self.w[:edge_idx]) * self.config.delx
        self.volume_history.append(volume)
        
        # Store current length
        current_length = edge_idx * self.config.delx
        self.length_history.append(current_length)
        
        # Check if we should evaluate for steady state (based on response time)
        if hasattr(self, 'last_evaluation_time'):
            time_since_last_eval = current_time - self.last_evaluation_time
            try:
                # Only evaluate every tau years
                if time_since_last_eval < self.tau_history[-1]:
                    return
            except ValueError:
                # Glacier is completely melted - consider it immediately at steady state
                # Skip the timing check and proceed directly to steady state evaluation
                pass
        else:
            # First evaluation - set baseline
            self.last_evaluation_time = current_time
            return
        
        # Check for steady state
        model_state = {
            'time_history': self.time_history,
            'volume_history': self.volume_history,
            'length_history': self.length_history
        }
        
        # Special case: completely melted glacier is considered at steady state
        glacier_melted = edge_idx <= 0
        
        if glacier_melted or self.steady_state_detector(model_state):
            if not self.steady_state_achieved:
                print(f"Steady state achieved at t={current_time:.1f} years")
                self.steady_state_achieved = True
                
                # Calculate final cost
                final_model_state = {
                    'edge': edge_idx,
                    'delx': self.config.delx,
                    'b': b,
                    'ela': self.ela[-1],
                    'F': self.F[-1],
                    'h': h,
                    'volume': volume,
                    'area': np.sum(self.w[:edge_idx]) * self.config.delx if edge_idx > 0 else 0.0
                }
                
                self.optimization_cost = self.cost_function(final_model_state, self.optimization_targets)
                
                print(f"Final cost: {self.optimization_cost:.1f}")
                
                # Early termination if cost is acceptable
                tolerance = self.target_matching.get('tolerance', 100)
                # Square the tolerance since cost function now returns squared error
                tolerance_squared = tolerance * tolerance
                if self.optimization_cost <= tolerance_squared:
                    # Report in terms of actual error (sqrt of cost)
                    actual_error = np.sqrt(self.optimization_cost)
                    print(f"Target achieved within tolerance ({tolerance}), actual error: {actual_error:.1f}")
                    # We could implement early termination here if needed
        
        # Update last evaluation time
        self.last_evaluation_time = current_time

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
        # Interpolate raw geometry to the model grid 'x' to save it as a data variable.
        # This avoids creating extra dimensions that interfere with sweep analysis tools.
        zb_gr_interp_func = interp1d(self.x_gr, self.zb_gr, bounds_error=False, fill_value="extrapolate")
        w_geom_interp_func = interp1d(self.x_gr, self.w_geom, bounds_error=False, fill_value="extrapolate")

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
            # Add resampled original geometry as data variables with dimension 'x'
            'zb_gr_resampled': (['x'], zb_gr_interp_func(self.x)),
            'w_geom_resampled': (['x'], w_geom_interp_func(self.x)),
        }
        
        # Add climate-specific variables if they exist
        if hasattr(self, 'T') and self.T is not None:
            data_vars['T'] = (['time'], self.T)
        if hasattr(self, 'pdd') and self.pdd is not None:
            data_vars['pdd'] = (['time', 'x'], self.pdd)
        
        config_dict = asdict(self.config)
        # Filter out None values, as they are not supported by netCDF attributes
        attrs = {k: v for k, v in config_dict.items() if v is not None}
        # Add forcing parameters to attributes for reproducibility and testing.
        # This is done generically to capture all forcing parameters.
        if hasattr(self.forcing, '__dict__'):
            for key, value in vars(self.forcing).items():
                if key.startswith('_'):
                    continue

                attr_key = f'forcing_{key}'
                if isinstance(value, (list, np.ndarray, tuple)):
                    # For array-like parameters, store a representation of their shape
                    # to prevent excessively long labels in sweep plots.
                    attrs[attr_key] = f"<array shape={str(np.array(value).shape)}>"
                elif value is not None:
                    attrs[attr_key] = value
        # Store the original geometry grid as numpy array attributes
        attrs['x_gr'] = self.x_gr
        attrs['zb_gr'] = self.zb_gr
        attrs['w_geom'] = self.w_geom

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': self.t,
                'x': self.x,
            },
            attrs=attrs
        )
        
        return ds

    def copy(self):
        return copy.deepcopy(self)


@nb.njit(fastmath={"contract"})
def space_loop(h, b, x, rho, g, nxs, delx, dzbdx, fd, fs, dwdx, w, delt, min_thick, n,k):
    Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
    Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
    dhdt = np.zeros(x.size)  # zero out thickness rate of change array
    
    # Pre-calculate powers of rho*g for performance
    rho_g_n = (rho * g) ** n
    rho_g_k = (rho * g) ** k
    
    dzdx = (dzbdx[:-1] + dzbdx[1:]) / 2  # slope at plus half a grid point
    # -----------------------------------------
    # begin loop over space
    # -----------------------------------------
    for j in range(0, nxs - 1):
        if j == 0:
            h_ave = (h[0] + h[1]) / 2
            dhdx = (h[1] - h[0]) / delx
            slope = dhdx + dzdx[j]
            
            # Separate flux calculation for deformation and sliding
            flux_d = -fd * rho_g_n * h_ave**(n+2) * abs(slope)**(n-1) * slope
            flux_s = -fs * rho_g_k * h_ave**k * abs(slope)**(k-1) * slope
            Qp[0] = flux_d + flux_s

            # Qm[0] = 0  # flux at minus half grid point
            dhdt[0] = b[0] - Qp[0] / delx - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
        elif (h[j] <= min_thick) & (h[j - 1] > min_thick):  # glacier toe condition
            # Qp[j] = 0
            h_ave = (h[j] + h[j-1]) / 2
            dhdx = (h[j] - h[j-1]) / delx
            slope = dhdx + dzdx[j - 1]
            
            flux_d = -fd * rho_g_n * h_ave**(n+2) * abs(slope)**(n-1) * slope
            flux_s = -fs * rho_g_k * h_ave**k * abs(slope)**(k-1) * slope
            Qm[j] = flux_d + flux_s

            dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
        elif (h[j] <= min_thick) & (h[j - 1] <= min_thick):  # beyond glacier toe - no glacier flux
            dhdt[j] = b[j]
            Qp[j] = 0
            Qm[j] = 0
        else:  # within the glacier
            # Flux at j + 1/2
            h_ave = (h[j + 1] + h[j]) / 2
            dhdx = (h[j + 1] - h[j]) / delx
            slope = dhdx + dzdx[j]
            
            flux_d = -fd * rho_g_n * h_ave**(n+2) * abs(slope)**(n-1) * slope
            flux_s = -fs * rho_g_k * h_ave**k * abs(slope)**(k-1) * slope
            Qp[j] = flux_d + flux_s

            # Flux at j - 1/2
            h_ave = (h[j - 1] + h[j]) / 2
            dhdx = (h[j] - h[j - 1]) / delx
            slope = dhdx + dzdx[j - 1]
            
            flux_d = -fd * rho_g_n * h_ave**(n+2) * abs(slope)**(n-1) * slope
            flux_s = -fs * rho_g_k * h_ave**k * abs(slope)**(k-1) * slope
            Qm[j] = flux_d + flux_s

            dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
    # ----------------------------------------
    # end loop over space
    # ----------------------------------------
    #dhdt[nxs] = 0  # enforce no change at boundary
    h = np.core.umath.maximum(h + dhdt * delt, 0)
    
    # More robust edge detection
    edge = nxs      # Default to end of domain
    for i in range(nxs):
        if h[i] < min_thick:
            edge = i
            break
    
    F = Qm + Qp
    return h, edge, F
