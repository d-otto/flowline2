"""
Linear glacier model for length change calculations.

Based on the 3-stage linear glacier model for calculating length perturbations
from mass balance changes.
"""

import numpy as np
import numba as nb


class LinearModel:
    """
    Linear glacier model for calculating length changes from mass balance perturbations.
    
    This implements the 3-stage linear glacier model equation for length perturbations.
    """
    
    def __init__(self, L_bar, H, tau, dt=1.0):
        """
        Initialize linear model parameters.
        
        Parameters
        ----------
        L_bar : float
            Reference glacier length (m)
        H : float
            Characteristic ice thickness near terminus (m)
        tau : float
            Glacier response time (years)
        dt : float, optional
            Time step (years), default 1.0
        """
        self.L_bar = L_bar
        self.H = H
        self.tau = tau
        self.dt = dt
        
        # Fixed coefficients for 3-stage model
        self.eps = 1 / np.sqrt(3)
        self.K = 1 - self.dt / (self.eps * self.tau)
        self.beta = self.L_bar / self.H
    
    def calc_length_change_for_mass_balance(self, mass_balance_perturbation):
        """
        Calculate length change from mass balance perturbation time series.
        
        Parameters
        ----------
        mass_balance_perturbation : array_like
            Time series of mass balance perturbations (m/year)
            
        Returns
        -------
        array_like
            Length change time series (m)
        """
        if np.isscalar(mass_balance_perturbation):
            # Single value - create short time series
            bt_p = np.array([0, 0, 0, mass_balance_perturbation])
        else:
            bt_p = np.asarray(mass_balance_perturbation)
        
        # Initialize length change array
        length_change = np.zeros_like(bt_p, dtype=float)
        
        # Use numba-compiled function for performance
        length_change = _calc_length_change_numba(
            length_change, self.K, self.dt, self.tau, self.eps, self.beta, bt_p
        )
        
        return length_change
    
    def steady_state_length_change(self, mass_balance_change):
        """
        Calculate steady-state length change for a given mass balance change.
        
        Parameters
        ----------
        mass_balance_change : float
            Mass balance change (m/year)
            
        Returns
        -------
        float
            Steady-state length change (m)
        """
        return self.tau * self.beta * mass_balance_change


@nb.njit()
def _calc_length_change_numba(length_change, K, dt, tau, eps, beta, bt_p):
    """
    Numba-compiled function for calculating length change time series.
    
    This implements the 3-stage linear glacier model equation:
    L(i) = 3*K*L(i-1) - 3*K^2*L(i-2) + K^3*L(i-3) + dt*beta/eps*(dt/(eps*tau))^2*bt_p(i-3)
    """
    for i in range(len(length_change)):
        if i <= 2:
            # First few time steps - only use previous terms
            if i >= 1:
                length_change[i] += 3 * K * length_change[i - 1]
            if i >= 2:
                length_change[i] -= 3 * K**2 * length_change[i - 2]
        else:
            # Full equation with mass balance forcing
            length_change[i] = (
                3 * K * length_change[i - 1]
                - 3 * K**2 * length_change[i - 2]
                + K**3 * length_change[i - 3]
                + dt * beta / eps * (dt / (eps * tau))**2 * bt_p[i - 3]
            )
    
    return length_change


def calc_Leq(A, w, bt, db, L=None):
    """
    Calculate equilibrium length change for mass balance perturbation.
    
    Parameters
    ----------
    A : float
        Glacier area (m²)
    w : float or array
        Glacier width (m), will be averaged if array
    bt : float
        Terminus mass balance (m/yr)
    db : float
        Mass balance change (m/yr)
    L : float, optional
        Reference length (m), not used in current implementation
        
    Returns
    -------
    float
        Equilibrium length change (m)
    """
    if np.ndim(w) != 0:
        w = np.mean(w)
    return A / w * -db / bt