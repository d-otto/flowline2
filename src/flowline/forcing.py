from abc import ABC, abstractmethod
import numpy as np
import scipy as sci
import numba as nb

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
        elif T[i] >= Tamp:
            pdd[i] = T[i]
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
        # Clip year_idx to prevent index out of bounds on the last step
        year_idx = min(year_idx, len(self.Pp) - 1)
        
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
        if np.isscalar(self.b0):
            b = np.full_like(x, self.b0, dtype=float)
        else:
            b = np.array(self.b0, dtype=float)
        
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


