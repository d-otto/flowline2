import numpy as np
import xarray as xr
import numba as nb
from scipy.optimize import fsolve

from flowline.diagnostics import calc_ela, calc_mass_balance



def solve_ela_for_parameter(target_variable, target_value, P0, T0=None, gamma=None, mu=None, initial_guess=10):
    """
    Solves the ELA equation for a single unknown variable using a numerical solver.

    This function finds the value of one parameter (e.g., T0) that results in a
    specified Equilibrium Line Altitude (ELA), holding other parameters constant.

    Parameters
    ----------
    target_variable : str
        The name of the variable to solve for. Must be one of 'T0', 'gamma', 'mu', 'P0'.
    target_value : float
        The desired output value for the ELA.
    P0, T0, gamma, mu : float, optional
        Known values for the ELA equation parameters. The parameter corresponding
        to `target_variable` should be set to None.
    initial_guess : float, optional
        Initial guess for the solver.

    Returns
    -------
    float
        The calculated value of the target_variable that satisfies the ELA equation.
    """
    # Define the root function for the solver. It calculates `calc_ela(...) - target_value`.
    def root_function(x):
        params = {'P0': P0, 'T0': T0, 'gamma': gamma, 'mu': mu}
        params[target_variable] = x[0]
        # The equation to solve is: calc_ela(...) - target_ela = 0
        return calc_ela(P0=params['P0'], T0=params['T0'], gamma=params['gamma'], mu=params['mu']) - target_value

    # Solve for the root
    solution, = fsolve(root_function, [initial_guess])
    return solution

def create_parameter_sweep( 
                         elev_range=(0, 2000, 25),
                         mu_range=(0.2, 1.4, 0.025), 
                         gamma_range=(2, 20, 0.1),
                         T0_range=(5, 20, 0.05),
                         P0=1000):  # Fixed winter accumulation
    """
    Create xarray dataset with parameter sweep
    
    Parameters:
    -----------
    elev_range : tuple
        (min, max, step) for elevation in meters
    mu_range : tuple  
        (min, max, step) for melt factor (m/°C/yr)
    gamma_range : tuple
        (min, max, step) for lapse rate in C/km
    T0_range : tuple
        (min, max, step) for sea level temperature in C
    P0 : float
        Winter accumulation (mm w.e.) - kept constant
    """
    
    # Create coordinate arrays
    elevation = np.arange(*elev_range)
    mu_vals = np.arange(*mu_range)
    gamma_vals = np.arange(*gamma_range) 
    T0_vals = np.arange(*T0_range)
    
    # Create coordinate meshgrid for xarray
    coords = {
        'elevation': elevation,
        'mu': mu_vals,
        'gamma': gamma_vals,
        'T0': T0_vals
    }
    
    # Vectorize calculations using NumPy broadcasting to avoid slow Python loops.
    
    # Reshape coordinate arrays for mass balance calculation.
    # The new shapes align with the dimensions of the final mass_balance array:
    # (elevation, mu, gamma, T0)
    h_grid        = elevation.reshape(-1, 1, 1, 1)
    mu_grid_mb    = mu_vals.reshape(1, -1, 1, 1)
    gamma_grid_mb = gamma_vals.reshape(1, 1, -1, 1)
    T0_grid_mb    = T0_vals.reshape(1, 1, 1, -1)

    # Calculate mass balance for all parameter combinations at once
    mass_balance_data = calc_mass_balance(h_grid, P0, T0_grid_mb, gamma_grid_mb, mu_grid_mb)
    
    # Reshape coordinate arrays for ELA calculation.
    # The new shapes align with the dimensions of the final ELA array:
    # (mu, gamma, T0)
    mu_grid_ela    = mu_vals.reshape(-1, 1, 1)
    gamma_grid_ela = gamma_vals.reshape(1, -1, 1)
    T0_grid_ela    = T0_vals.reshape(1, 1, -1)
    
    # Calculate ELA for all parameter combinations at once
    ela_data = calc_ela(P0, T0_grid_ela, gamma_grid_ela, mu_grid_ela)
    
    # Create xarray datasets
    dataset = xr.Dataset({
        'mass_balance': (['elevation', 'mu', 'gamma', 'T0'], mass_balance_data),
        'P0': P0
    }, coords=coords)
    
    ela_dataset = xr.Dataset({
        'ELA': (['mu', 'gamma', 'T0'], ela_data),
        'P0': P0
    }, coords={k: v for k, v in coords.items() if k != 'elevation'})
    
    return dataset, ela_dataset
