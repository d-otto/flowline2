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
import gm


class flowline2d:
    def __init__(
        self,
        x_gr,
        zb_gr,
        w_geom,
        mode='TP',
        sigT=1,
        sigP=1,
        sigb=1,
        T0=None,
        P0=None,
        T=None,
        P=None,
        x_init=None,
        x_geom=None,
        h_geom=None,
        profile=None,
        t_stab=None,
        temp=None,
        gamma=6.5e-3,
        dpdz=None,
        mu=0.65,
        g=9.81,
        rho=916.8,
        fd=1.9e-24,
        fs=5.7e-20,
        delx=50,
        delt=0.0125 / 8,
        ts=0,
        tf=2025,
        dt_plot=100,
        rt_plot=False,
        xlim0=None,
        min_thick=1,
        hmb=True,
        T2melt=None,
        pdd_Tamp=None,
        pdd_beta=None,
        bz=None,
        bx=None,
        bp=None,
        bal=None,
        b0=None,
        deltout=1,
        n=3,
        k=3,
    ):
        """2d flowline model

        This module demonstrates documentation as specified by the `NumPy
        Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
        are created with a section header followed by an underline of equal length.

        Parameters
        ----------
        x_gr : array-like
            x dimension of bed topography; m
        zb_gr : array-like
            z dimension of bed topography; m
        x_geom : array-like
            x dimension of glacier width; m
        w_geom : array-like
            width of glacier along flowline; m
        x_init : array-like
            x dimension of initial thickness profile; m
        h_init : array-like
            initial ice thickness profile; m
        T : array-like
            random state for temperature
        P : array-like
            random state for precipitation
        rho : float
            Ice density kg/m^3
        g : float
            Gravity m/s^2
        mu : float
            Melt rate m/yr/degC
        n : float
            Glenn's flow law parameter. Default n = 3
        gamma : float
            Temperature lapse rate degC/km
        sigT : float
            Temperature standard deviation degC
        sigP : float
            Precipitation standard deviation m/yr
        T0 : float
            Baseline temperature degC
        fd : float
            Deformation parameter Pa^-3 s^-2
        fs : float
            Sliding parameter Pa^-3 s^-1 m^2
        delx : int
            Grid spacing in m
        delt : float
            Time step in yrs suitable for 200m yr^-1
        ts : float
            Starting time yr
        tf : float
            Ending time yr
        dt_plot : int
            Plotting interval yr
        rt_plot : bool
            Whether there should be real time plotting while the model is running.
        xlim0 : float
            Left limit for plots (yrs)


        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        # # How well do the equilibrium responses match?
        # # How well does the timescale match?
        # # What is the distribution of trends?
        # # How does the run length work? Does it follow a Poisson process?
        # # Is the dynamical model consistent with a Gaussian pdf?

        self.init_args = locals().copy()

        # #-----------------
        # #define parameters
        # #-----------------

        xmx = np.max(delx * np.floor(x_geom / delx))  # round to neaest delx
        x = np.arange(0, xmx, delx)  # x array
        nxs = len(x)

        fd = fd * np.pi * 1e7
        fs = fs * np.pi * 1e7  # convert from seconds to years

        # ---------------------------------
        # different glacier bed geometries
        # ---------------------------------

        zb = interp1d(x_gr, zb_gr)  # elevation of the bed
        zb = zb(x)
        w = interp1d(x_gr, w_geom)  # width @ surface
        w = w(x)
        dzbdx = np.gradient(zb, x)  # slope of glacer bed geometries.
        dwdx = np.gradient(w, x)  # change in width between grid cells

        # geometry errors/warnings
        if any(dzbdx == 0):
            logging.warning(f'Bed slope is zero at {(dzbdx == 0).argmax()}.')
        if any(dzbdx[0:2] > 0):
            logging.warning('The slope of the bed at the top of the glacier is positive. This may cause instabilities.')

        # initialize climate forcing
        # sorry the inputs are kind of a cluster
        self.nts = round(np.floor((tf - ts) / delt))  # number of time steps
        nyrs = tf - ts
        if T is None:  # temperature
            T = np.zeros(nyrs)
        if P is None:  # precip
            P = np.zeros(nyrs)
        self.Tp = sigT * T  # Temperature perturbation
        self.Pp = sigP * P  # Precip perturbation
        if bp is None:   # mass balance perturbation (if T & P are not set)
            bp = np.zeros(nyrs)
        if bal is None:  # mass balance trend added to bp
            self.bp = bp * sigb
        else:
            self.bp = bp * sigb + bal
        if temp is None:  # temperature trend added to Tp
            temp = np.zeros(nyrs)
        if t_stab:  # number of years of stable climate at the beginning of the simulation
            self.Tp.iloc[:t_stab] = 0
            self.Pp.iloc[:t_stab] = 0
            temp.iloc[:t_stab] = 0
        self.ts = ts  # time start
        self.tf = tf  # time finish
        self.temp = temp
        self.bal = bal
        self.T = T
        self.P = P
        self.t_stab = t_stab
        self.pdd_Tamp = pdd_Tamp  # amplitude for positive degree days
        self.pdd_beta = pdd_beta  # beta for positive degree days
        if mode == 'b':  # if inputs are in terms of mass balance (aka no T & P provided)
            if bp is None:
                bp = np.zeros(nyrs)
            self.bp = bp
            if bal is None:
                bal = np.zeros(nyrs)
            self.bal = bal
            if sigb is None:
                sigb = 1
            self.sigb = sigb
        # constants
        self.sigT = sigT
        self.sigP = sigP
        self.P0 = P0  # baseline accumulation (for calibration)
        self.T0 = T0  # baseline temp (for calibration)
        self.mu = mu  # melt factor
        self.gamma = gamma  # lapse rate
        self.rho = rho  # density
        self.g = g  # gravity
        self.min_thick = min_thick  # minimum thickness for ice at the terminus
        self.nxs = nxs
        self.delt = delt
        self.dzbdx = dzbdx
        self.dwdx = dwdx
        self.fd = fd
        self.fs = fs
        self.n = n
        self.k = k
        self.x = x
        self.zb = zb
        self.nxs = nxs
        self.delx = delx
        self.w = w
        self.b0 = b0  # baseline mass balance (for calibration)
        try:
            self.x_geom = x_geom
            self.h_geom = h_geom
        except:
            pass
        self.profile = profile  # loaded geometry
        self.deltout = deltout  # frequency to save output

        # look up table
        self.bz = bz
        self.bx = bx
        if dpdz is None:
            dpdz = np.zeros(int(zb.max()) + 500)
        self.dpdz = dpdz

        # option flags
        self.mode = mode
        self.hmb = hmb  # height mass balance
        self.dt_plot = dt_plot
        self.rt_plot = rt_plot
        
        # dynamic flags? could be callable or string or None
        self.T2melt = T2melt

        # runtime flags
        self.no_error = True

    def run(self, **kwargs):
        # run the model depending on the mode
        
        if kwargs:  # change any arguments that were provided in the run call
            self.__dict__.update(kwargs)

        self.load_profile()  # load glacier geometry

        # call the appropriate run function
        if self.mode == 'TP':
            return self._run_TP()
        elif self.mode == 'b':
            return self._run_b()

    def _run_TP(self):
        # output
        nouts = int((self.nts * self.delt) // 1)
        self.edge_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.edge = np.full(nouts, fill_value=np.nan, dtype="float")
        self.t = np.full(nouts, fill_value=np.nan, dtype="float")
        self.T = np.full(nouts, fill_value=np.nan, dtype="float")
        self.gwb = np.full(nouts, fill_value=np.nan, dtype="float")
        self.ela = np.full(nouts, fill_value=np.nan, dtype="float")
        self.area = np.full(nouts, fill_value=np.nan, dtype="float")
        self.h = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.b = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.melt = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.ela_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.F = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.P = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.pdd = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")

        yr = self.ts - 1  # - 1 because we start the time loop by incrementing the year
        idx_out = 0

        if self.rt_plot:
            self.fig, self.ax = self._init_plot()

        h = self.h0  # define initial height
        for i in tqdm(
            range(0, self.nts),
            unit_scale=self.delt,
            unit="yrs",
            bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
            ascii=True,
            ncols=100,
        ):
            t = self.delt * i  # time in years

            # UPDATE climate every year
            if t == t // 1:
                yr = yr + 1

                # effective height
                # without hmb this has to stay fixed between runs even while the profile changes
                if self.hmb:
                    h_eff = self.zb + h
                else:
                    h_eff = self.zb

                P = (self.P0 + self.Pp[yr - self.ts]) * np.ones(self.x.size)# + self.dpdz[h_eff.astype(int)]
                T_wk = (
                    (self.T0 + self.Tp[yr - self.ts]) * np.ones(self.x.size) + self.temp[yr - self.ts] - self.gamma * h_eff
                )  # add temperature forcing
                if callable(self.T2melt):
                    melt = self.T2melt(T_wk)
                elif self.T2melt == 'pdd':
                    pdd = calc_pdd(T_wk, self.pdd_Tamp)
                    melt = np.maximum(0, pdd * self.mu)
                else:
                    melt = np.maximum(0, T_wk * self.mu)  # this is apparently faster than clip for numpy 1.17
                
                b = P - melt

            # loop over space (solve SIA)
            # this is the entire model, really
            h, edge_idx, F = space_loop(
                h,
                b,
                self.x,
                self.rho,
                self.g,
                self.nxs,
                self.delx,
                self.dzbdx,
                self.fd,
                self.fs,
                self.dwdx,
                self.w,
                self.delt,
                self.min_thick,
                self.n,
                self.k,
            )

            if t / self.deltout == np.floor(t / self.deltout):
                # Save outputs
                area = np.sum(self.w[:edge_idx]) * self.delx
                bal = b * self.w * self.delx  # mass added in a given cell units are m^3 yr^-1
                # bal[edge+1] =
                self.gwb[idx_out] = bal[
                    :edge_idx
                ].sum()  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)

                self.T[idx_out] = self.T0 + self.Tp[yr - self.ts] + self.temp[yr - self.ts]  # input temperature
                self.P[idx_out, :] = P
                self.melt[idx_out, :] = melt
                self.t[idx_out] = t + self.ts
                self.edge_idx[idx_out] = edge_idx
                self.edge[idx_out] = edge_idx * self.delx
                self.h[idx_out, :] = h
                self.area[idx_out] = area
                try:
                    ela_idx = np.abs(b[:edge_idx]).argmin()
                except:  # if the ela is above the glacier
                    ela_idx = 0
                self.ela_idx[idx_out] = ela_idx
                self.ela[idx_out] = self.zb[ela_idx] + h[ela_idx]
                self.b[idx_out, :] = b
                if self.T2melt == 'pdd':
                    self.pdd[idx_out, :] = pdd
                # b_out[idx_out, edge+1:] = np.nan

                self.F[idx_out, :] = F
                idx_out = idx_out + 1

                if self.rt_plot:
                    self._rt_plot(t)

                # -----------------------------------------
                # end loop over time
                # -----------------------------------------

        if np.isnan(self.h[-1, 0]):
            self.no_error = False
        else:
            self.no_error = True

        return copy.deepcopy(self)

    def _run_b(self):
        # output
        nouts = int((self.nts * self.delt) // 1)
        self.edge_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.edge = np.full(nouts, fill_value=np.nan, dtype="float")
        self.t = np.full(nouts, fill_value=np.nan, dtype="float")
        self.gwb = np.full(nouts, fill_value=np.nan, dtype="float")
        self.ela = np.full(nouts, fill_value=np.nan, dtype="float")
        self.area = np.full(nouts, fill_value=np.nan, dtype="float")
        self.h = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.b = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.melt = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.ela_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.F = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")

        yr = self.ts - 1  # - 1 because we start the time loop by incrementing the year
        idx_out = 0

        if self.rt_plot:
            self.fig, self.ax = self._init_plot()

        h = self.h0  # define initial height
        for i in tqdm(
            range(0, self.nts),
            unit_scale=self.delt,
            unit="yrs",
            bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
            ascii=True,
            ncols=100,
        ):
            t = self.delt * i  # time in years

            # UPDATE climate every year
            if t == t // 1:
                yr = yr + 1

                # effective height
                # without hmb this has to stay fixed between runs even while the profile changes
                if self.hmb:
                    h_eff = self.zb + h
                else:
                    h_eff = self.zb + h[0]

                # set forcing for the year
                if self.bz is not None:
                    b = self.b0 + self.bp[yr - self.ts] * self.sigb + self.bal[yr - self.ts] + self.bz[h_eff.astype(int)]
                else:
                    b = self.b0 + self.bp[yr - self.ts] * self.sigb + self.bal[yr - self.ts] + self.bx[self.x.astype(int)]


            # loop over space (solve SIA)
            # this is the entire model, really
            h, edge_idx, F = space_loop(
                h,
                b,
                self.x,
                self.rho,
                self.g,
                self.nxs,
                self.delx,
                self.dzbdx,
                self.fd,
                self.fs,
                self.dwdx,
                self.w,
                self.delt,
                self.min_thick,
                self.n,
                self.k,
            )

            # record output
            if t / self.deltout == np.floor(t / self.deltout):
                # Save outputs
                area = np.sum(self.w[:edge_idx]) * self.delx
                bal = b * self.w * self.delx  # mass added in a given cell units are m^3 yr^-1
                # bal[edge+1] =
                self.gwb[idx_out] = bal[
                    :edge_idx
                ].sum()  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)
                self.t[idx_out] = t + self.ts
                self.edge_idx[idx_out] = edge_idx
                self.edge[idx_out] = edge_idx * self.delx
                self.h[idx_out, :] = h
                self.area[idx_out] = area
                try:
                    ela_idx = np.abs(b[:edge_idx]).argmin()
                except:  # if the ela is above the glacier
                    ela_idx = 0
                self.ela_idx[idx_out] = ela_idx
                self.ela[idx_out] = self.zb[ela_idx] + h[ela_idx]
                self.b[idx_out, :] = b
                # b_out[idx_out, edge+1:] = np.nan

                self.F[idx_out, :] = F
                idx_out = idx_out + 1

                if self.rt_plot:
                    self._rt_plot(t)

                # -----------------------------------------
                # end loop over time
                # -----------------------------------------

        if np.isnan(self.h[-1, 0]):
            self.no_error = False
        else:
            self.no_error = True

        return copy.deepcopy(self)

    def load_profile(self):
        # load/fill in initial values for glacier thickness from previous run or default
        # by the end of this section there will be values for x0 and h0
        try:  # if an initial profile was provided
            h0 = np.array(self.profile.h[-1, :])
            x0 = np.array(self.profile.x)
        except:  # initial profile is not flowline2d object
            try:  # try treating self.profile as file path to flowline2d object
                with open(self.profile, 'rb') as f:
                    last_run = dill.load(f)
                h0 = np.array(last_run.h[-1, :])  # take the final year of the simulation
                x0 = np.array(last_run.x)
                logging.info(f"Successfully loaded profile: {self.profile}")

            except Exception as error:  # use default values
                logging.debug("Exception on profile loading: ", error)
                logging.info(f"Did not load profile. Using default values for x0 and h0.")
                x0 = self.x_geom
                h0 = self.h_geom

        # values for x0 and h0 have been set
        # interpolate x0 and h0 to model grid
        try:
            h0 = interp1d(x0, h0, "linear", bounds_error=True)
            h0 = h0(self.x)
        except:
            logging.warning(
                f"A value in x exceeds x0 for interpolation of h0 to the model grid. Proceeding with extrapolation. x0.max() = {x0.max()}, x.max() = {self.x.max()}"
            )
            h0 = interp1d(x0, h0, "linear", fill_value="extrapolate")
            h0 = h0(self.x)
        self.h0 = h0
        self.x_geom = x0
        self.h_geom = h0  # to fix hmb drift

    def plot_full(self, xlim0=None, smooth=20):
        """This is a docstring

        This is the longer portion of the docstring.

        Parameters
        ----------------
        xlim0 : float
            left x-limit for figure (years)

        Returns
        ----------------
        fig : Figure
            It's a figure??

        """
        if xlim0 is None:
            xlim0 = self.ts

        pad = 20
        pedge = int(self.edge_idx[-1]) + pad
        self.pedge = pedge
        x1 = self.x[:pedge]
        z0 = self.zb[:pedge]
        z1 = z0 + self.h[-1, :pedge]

        fig, ax = self._init_plot()
        ax[0, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.area / 1e6, smooth, mode="mirror"),
            c="black",
            label=f"MA-{smooth}",
        )
        poly1 = ax[0, 1].fill_between(
            x1 / 1000,
            z0,
            z1,
            fc="lightblue",
            ec="lightblue",
            label=f"{self.tf} profile",
        )
        ax[0, 1].plot(
            x1 / 1000,
            z0,
            c="black",
            lw=2,
        )
        ax[0, 2].hist(
            self.gwb / self.area,
            bins=100,
            density=True,
        )
        ax[0, 2].axvline(x=(self.gwb / self.area).mean(), ls="--", lw=2, c="black", label="Mean")
        ax[0, 2].annotate(
            f"b_s = {np.std(self.gwb / self.area):0.4f}\n" f"mean = {np.mean(self.gwb/self.area):0.4f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        ax[1, 2].hist(
            self.edge / 1000,
            bins=30,
            density=True,
        )
        ax[1, 2].axvline(x=(self.edge / 1000).mean(), ls="--", lw=2, c="black", label="Mean")
        ax[1, 2].annotate(
            f"$\sigma_l$ = {np.std(self.edge / 1000):0.4f}\n" f"mean = {np.mean(self.edge):0.4f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        ax[0, 0].set_xlim(xlim0, self.tf)
        ax[1, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.ela, smooth, mode="mirror"),
            c="black",
            label=f"MA-{smooth}",
        )
        ax[1, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].set_xlim(0, x1.max() / 1000 * 1.1)
        # ax[2, 0].plot(self.t, self.T, c="blue", lw=0.25, alpha=0.25)
        ax[2, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.T, smooth, mode="mirror"),
            c="black",
            lw=1,
            alpha=0.5,
            label=f"MA-{smooth}",
        )
        ax[2, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.T, 300, mode="mirror"),
            c="red",
            lw=1,
            label="MA-300",
        )
        ax[2, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].plot(
            self.t,
            self.edge / 1000,
            c="black",
            lw=2,
            label=f"Length",
        )
        ax[2, 1].set_xlim(xlim0, self.tf)
        ax[2, 2].scatter(
            scipy.ndimage.uniform_filter1d(self.edge / 1000, 100, mode="mirror"),
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, 100, mode="mirror"),
            c=self.t,
            cmap="viridis",
            s=2,
            label="MA-100",
        )
        # ax[2, 2].set_xlim(xlim0, self.tf)
        # ax[3, 0].plot(self.t, self.gwb / self.area, c="blue", lw=0.25)
        ax[3, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, smooth, mode="mirror"),
            c="black",
            lw=1,
            alpha=0.5,
            label=f"MA-{smooth}",
        )
        ax[3, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, 300, mode="mirror"),
            c="red",
            lw=1,
            label="MA-300",
        )
        ax[3, 0].set_xlim(xlim0, self.tf)
        ax[3, 1].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(
                np.cumsum(self.gwb / self.area),
                smooth,
                mode="mirror",
            ),
            c="blue",
            lw=2,
            label=f"MA-{smooth}",
        )
        ax[3, 1].set_xlim(xlim0, self.tf)
        scat = ax[3, 2].scatter(
            scipy.ndimage.uniform_filter1d(self.h.mean(axis=1), smooth, mode="mirror"),
            scipy.ndimage.uniform_filter1d(self.edge / 1000, smooth, mode="mirror"),
            c=self.t,
            cmap="viridis",
            s=2,
            label=f"MA-{smooth}",
        )
        fig.colorbar(scat, ax=ax[2:, 2], label="Year")
        for axis in ax.ravel():
            try:
                axis.legend(loc="upper left")
            except:
                pass

        return fig, ax

    def plot(self, smooth=1):
        def sm(d):
            # todo: switch to butterworth filter?
            return pd.Series(d).rolling(smooth).mean()

        if smooth > 1:
            smooth_label = f'MA-{smooth}'
        else:
            smooth_label = ''
        fig, ax = plt.subplots(3, 2, layout='constrained', figsize=(8,6), dpi=200)
        pad = 20
        pedge = int(self.edge_idx[-1]) + pad
        self.pedge = pedge
        x1 = self.x[:pedge]
        z0 = self.zb[:pedge]
        z1 = z0 + self.h[-1, :pedge]
        poly1 = ax[0, 0].fill_between(
            x1 / 1000,
            z0,
            z1,
            fc="lightblue",
            ec="lightblue",
            label=f"yr={self.tf} profile",
        )
        ax[0, 0].plot(
            x1 / 1000,
            z0,
            c="black",
            lw=2,
        )
        
        
        if self.mode == 'b':
            ax[0,1].plot(
                self.t,
                sm(self.bp),
                c="blue",
                lw=1,
                label=f"b_anom {smooth_label}",
            )
        else:
            ax[0,1].plot(
                self.t,
                sm(self.T),
                c="blue",
                lw=1,
                label=f"T {smooth_label}",
            )
        #ax01b = ax[0, 1].twinx()
        ax[0,1].plot(self.t, sm(self.gwb / self.area), label=f'Sp. MB {smooth_label}', c='black', lw=1)
        #ax[0,1].plot([None], [None], label=f'Sp. MB {smooth_label}', c='black', lw=1, ls=':')  # just for legend

        ax[1, 0].plot(self.t, self.h.max(axis=1), c='limegreen', label=f"Max H {smooth_label}", lw=1)
        ax10b = ax[1, 0].twinx()
        ax10b.plot(
            self.t,
            sm(self.ela),
            c='blue',
            ls='--',
            lw=0.5,
            label=f"ELA",
        )
        ax[1, 0].plot(  # just for the legend
            [None],
            [None],
            c='blue',
            ls='--',
            lw=1,
            label=f"ELA {smooth_label}",
        )

        ax[1, 1].plot(
            self.t,
            self.edge / 1000,
            c="black",
            lw=1,
            label=f"Length",
        )
        ax11b = ax[1, 1].twinx()
        ax11b.plot(
            self.t,
            self.area / 1e6,
            c='red',
            ls='--',
            lw=1,
        )
        ax[1, 1].plot(  # just for the legend
            [None],
            [None],
            c='red',
            ls='--',
            lw=1,
            label=f"Area",
        )
        ax[2, 0].plot(
            self.t,
            sm(self.b[np.arange(len(self.t)), self.edge_idx]),
            c='red',
            label='bt',
        )
        ax[2, 1].plot(
            self.t,
            sm([-self.h[i, j[0]:j[1]].mean() for i, j in enumerate(zip(self.edge_idx//2, self.edge_idx))] / self.b[np.arange(len(self.t)), self.edge_idx]),
            label='tau (-h_mean/bt)',
            color='black',
        )

        for i, axis in enumerate(ax.ravel()):
            axis.grid(which='both', axis='both', ls=':', c='grey')
            axis.legend(fontsize='small')

        ax[1, 0].set_ylabel('Max H [m]')
        # ax01b.grid(None)
        # ax01b.set_ylabel('Max H [m]')
        ax10b.grid(False)
        ax10b.set_ylabel('ELA [m]')
        ax11b.grid(False)
        ax11b.set_ylabel('Area [$km^2$]')

        # Re-arrange legends to last axis
        all_axes = fig.get_axes()
        for axis in all_axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
                all_axes[-1].add_artist(legend)

        return fig, ax

    def to_pickle(self, fp):
        with open(fp, "wb") as f:
            dill.dump(self, f)

        return None

    def to_pandas(self):
        d = dict(
            T=self.T,
            area=self.area,
            bal=self.gwb,
            edge=self.edge_idx,
            edge_m=self.edge,
            ela=self.ela,
        )
        df = pd.DataFrame(d, index=self.t)
        return df

    def to_xarray(self):
        ds = xr.Dataset(
            data_vars=dict(
                T=(["time"], self.T),
                P=(["time"], self.P),
                edge_idx=(["time"], self.edge_idx),
                edge=(["time"], self.edge),
                gwb=(["time"], self.gwb),
                b=(["time", "x"], self.b),
                ela=(["time"], self.ela),
                h=(["time", "x"], self.h),
                area=(["time"], self.area),
                w=(["x"], self.w),
                zb=(["x"], self.zb),
            ),
            coords=dict(
                time=self.t,
                x=self.x,
            ),
            attrs=dict(
                ts=self.ts,
                tf=self.tf,
                nxs=self.nxs,
                delx=self.delx,
                sigT=self.sigT,
                sigP=self.sigP,
                P0=self.P0,
                T0=self.T0,
                # Tref=self.Tref,
                # Pref=self.Pref,
                nrun=self.nrun,
                ref_period=self.ref_period,
            ),
        )
        return ds

    def copy(self):
        return copy.deepcopy(self)

    def calc_diag(res, t=(None, None)):
        tslice = slice(t[0], t[1])

        diag = pd.DataFrame(dtype=float, columns=['mean', 'std', 'mean_025', 'mean_975', 'std_025', 'std_975'])
        df = len(res.edge)
        b = res.gwb / res.area
        diag.loc['b', 'mean'] = b[tslice].mean()
        diag.loc['b', 'std'] = b[tslice].std()
        diag.loc['b', 'mean_025'], diag.loc['b', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['b', 'mean'], scale=diag.loc['b', 'std']
        )
        diag.loc['b', 'std_025'], diag.loc['b', 'std_975'] = gm.std_cinterval(b, 0.95)
        try:
            diag.loc['T', 'std'] = res.T[tslice].mean(axis=1).std()
            diag.loc['P', 'std'] = res.P[tslice].mean(axis=1).std()
        except:
            pass
        diag.loc['L', 'mean'] = res.edge[tslice].mean()
        diag.loc['L', 'std'] = res.edge[tslice].std()
        diag.loc['L', 'mean_025'], diag.loc['L', 'mean_975'] = sci.stats.t.interval(
            0.95, df, loc=diag.loc['L', 'mean'], scale=diag.loc['L', 'std']
        )
        diag.loc['L', 'std_025'], diag.loc['L', 'std_975'] = gm.std_cinterval(res.edge[tslice], 0.95)
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
        babl = np.array([res.b[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
        bacc = np.array([res.b[i, :j].mean() for i, j in enumerate(res.ela_idx[tslice])])
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
        bt = np.array([self.b[i, (self.ela_idx[i] - 10) : (self.edge_idx[i])].mean() for i in range(len(self.ela_idx))])
        tau = -H / bt
        return tau, H, bt

    def calc_tau2(self, t_idx):
        return (
            -self.h[np.arange(self.h.shape[0]), self.edge_idx - t_idx]
            / self.b[np.arange(self.b.shape[0]), self.edge_idx - t_idx]
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
        acx = gm.acf(res.edge, t)
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

    def calc_Leq(self):
        self.Leq = self.b.mean(axis=1) * self.edge[0] / self.b[np.arange(self.b.shape[0]), self.edge_idx - 30]
        return self.Leq

    def calc_return(self, L0=0):
        Leq = self.calc_Leq()
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

    def calc_tau_from_dLdt(self):
        sigdL = np.gradient(self.edge).std()
        sigL = self.edge.std()
        return sigL / sigdL

    def _init_plot(self):
        fig = plt.figure(figsize=(18, 10), dpi=100, layout="constrained")
        gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=(1, 1, 1, 1))
        ax = np.empty((4, 3), dtype="object")
        plt.show(
            block=False
        )  # for live plotting, though maybe block=True would wait for the plot to open before running?

        ax[0, 0] = fig.add_subplot(gs[0, 0])
        ax[0, 0].set_xlabel("Time (years)")
        ax[0, 0].set_ylabel("Glacier Area ($km^2$)")

        ax[0, 1] = fig.add_subplot(gs[0:2, 1])
        ax[0, 1].set_xlabel("Distance (km)")
        ax[0, 1].set_ylabel("Elevation (m)")

        ax[0, 2] = fig.add_subplot(gs[0, 2])
        ax[0, 2].set_xlabel("Mass balance (m)")
        ax[0, 2].set_ylabel("Probability density")

        ax[1, 0] = fig.add_subplot(gs[1, 0])
        ax[1, 0].set_xlabel("Time (years)")
        ax[1, 0].set_ylabel("Equilibrium Line Altitude (m)")

        ax[1, 2] = fig.add_subplot(gs[1, 2])
        ax[1, 2].set_xlabel("Length (km)")
        ax[1, 2].set_ylabel("Probability density")

        ax[2, 0] = fig.add_subplot(gs[2, 0])
        ax[2, 0].set_ylabel("T ($^o$C)")

        ax[2, 1] = fig.add_subplot(gs[2, 1])
        ax[2, 1].set_ylabel("L (km)")

        ax[2, 2] = fig.add_subplot(gs[2, 2])
        ax[2, 2].set_xlabel("L (km)")
        ax[2, 2].set_ylabel("Bal (m $yr^{-1}$)")

        ax[3, 0] = fig.add_subplot(gs[3, 0])
        ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
        ax[3, 0].set_xlabel("Time (years)")

        ax[3, 1] = fig.add_subplot(gs[3, 1])
        ax[3, 1].set_xlabel("Time (years)")
        ax[3, 1].set_ylabel("Cum. bal. (m)")

        ax[3, 2] = fig.add_subplot(gs[3, 2])
        ax[3, 2].set_xlabel("Mean thickness (m)")
        ax[3, 2].set_ylabel("Length (km)")

        for axis in ax.ravel():
            if axis is not None:  # this handles gridspec col/rowspans > 1
                axis.grid(axis="both", alpha=0.5)
                axis.set_axisbelow(True)
        plt.tight_layout()

        return fig, ax

    def _rt_plot(self, t, i):
        if (t / self.dt_plot == np.floor(t / self.dt_plot)) | (
            i == self.nts - 1
        ):  # force plotting on the last time step
            print("outputting")
            pad = 10
            x1 = self.x[: self.edge + pad]
            z0 = self.zb[: self.edge + pad]
            z1 = self.zb[: self.edge + pad] + self.h[: self.edge + pad]

            try:
                self.ax[0, 1].collections[0].remove()  # remove the glacier profile before redrawing
            except:
                pass
            poly = self.ax[0, 1].fill_between(x1 / 1000, z0, z1, fc="lightblue")
            self.ax[0, 1].plot(
                x1 / 1000,
                z0,
                c="black",
                lw=2,
            )
            self.ax[0, 0].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.area / 1e6, 20, mode="mirror"),
                c="black",
            )
            self.ax[1, 0].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.ela, 20, mode="mirror"),
                c="black",
            )
            self.ax[2, 0].plot(self.t, self.t, c="blue", lw=0.25)
            self.ax[2, 1].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.edge, 20, mode="mirror") / 1000,
                c="black",
                lw=2,
            )
            self.ax[3, 0].plot(self.t, self.gwb / self.area, c="blue", lw=0.25)
            self.ax[3, 1].plot(
                self.t,
                # scipy.ndimage.uniform_filter1d(
                #     np.cumsum(self.gwb / self.area), 20, mode="mirror"
                # ),
                np.cumsum(self.gwb / self.area) - np.cumsum((self.gwb / self.area).mean()),
                c="blue",
                lw=2,
            )

            # update the plot
            self.fig.canvas.flush_eveself.nts()
            self.fig.canvas.draw()

        if t == self.tf:
            plt.draw()


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
        elif (h[j] <= 0) & (h[j - 1] > 1):  # glacier toe condition
            # Qp[j] = 0
            h_ave = h[j - 1] / 2
            dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
            Qm[j] = (
                -rho_g_cu * (dhdx + dzdx[j - 1]) ** 3 * (fd * h_ave**(n+2) + fs * h_ave**k)
            )  # glacier toe qm
            dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
        elif (h[j] == 0) & (h[j - 1] < 1):  # beyond glacier toe - no glacier flux
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
    # edge = (
    #     #len(h) - np.searchsorted(h[::1], min_thick) - 1
    #     np.searchsorted(h, min_thick)
    # )  # very fast location of the terminus https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    edge = np.argmax(h<min_thick)
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
    

def fit_bprofile(bz, ba, z, P, T, gamma, mu, Tamp):

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
