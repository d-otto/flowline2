import numpy as np
import xarray as xr

def calc_ela(P0, T0, gamma, mu, h=None):
    """
    Calculate Equilibrium Line Altitude
    
    Parameters:
    -----------
    P0 : float/array
        Winter accumulation (mm w.e.)
    T0 : float/array  
        Melt-season temperature at sea level (°C)
    gamma : float/array
        Temperature lapse rate (°C/km)
    mu : float/array
        Melt factor (m/°C/yr)
    h : float/array, optional
        Elevation of glacier surface (m)
        
    Returns:
    --------
    ela : float/array
        Equilibrium Line Altitude (m)
    """
    # Convert gamma from C/km to C/m for calculations
    gamma_m = gamma / 1000
    
    # Convert P0 from mm to m for consistent units with mu
    P0_m = P0 / 1000
    
    # Adjust temperature for elevation if provided
    if h is not None:
        T0_adj = T0 - h * gamma_m
    else:
        T0_adj = T0
        
    # Calculate ELA (mu is in m/°C/yr, P0_m is in m/yr)
    ela = T0_adj / gamma_m - P0_m / (mu * gamma_m)
    return ela

def calc_mass_balance(h, P0, T0, gamma, mu):
    """
    Calculate mass balance at given elevation
    
    Parameters:
    -----------
    h : float/array
        Elevation (m)
    P0 : float/array
        Winter accumulation (mm w.e.)
    T0 : float/array
        Melt-season temperature at sea level (°C)
    gamma : float/array
        Temperature lapse rate (°C/km)
    mu : float/array
        Melt factor (m/°C/yr)
        
    Returns:
    --------
    mass_balance : float/array
        Annual mass balance (m w.e./yr)
    """
    gamma_m = gamma / 1000  # Convert C/km to C/m
    T_h = T0 - h * gamma_m  # Temperature at elevation h
    
    # Convert P0 from mm to m for consistent units
    P0_m = P0 / 1000
    
    # Simple mass balance model: accumulation - melt
    # Melt only occurs when temperature > 0
    melt = np.maximum(0, mu * T_h)  # mu is in m/°C/yr
    mass_balance = P0_m - melt  # Both in m w.e./yr
    
    return mass_balance

def create_parameter_sweep( 
                         elev_range=(0, 3000, 50),
                         mu_range=(0.2, 1.5, 0.05), 
                         gamma_range=(4, 10, 0.25),
                         T0_range=(5, 20, 0.1),
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
    
    # Initialize data arrays
    mass_balance_data = np.zeros((len(elevation), len(mu_vals), 
                                len(gamma_vals), len(T0_vals)))
    ela_data = np.zeros((len(mu_vals), len(gamma_vals), len(T0_vals)))
    
    # Calculate mass balance for all parameter combinations
    for i, h in enumerate(elevation):
        for j, mu in enumerate(mu_vals):
            for k, gamma in enumerate(gamma_vals):
                for l, T0 in enumerate(T0_vals):
                    mass_balance_data[i,j,k,l] = calc_mass_balance(
                        h, P0, T0, gamma, mu)
    
    # Calculate ELA for all parameter combinations (excluding elevation)
    for j, mu in enumerate(mu_vals):
        for k, gamma in enumerate(gamma_vals):
            for l, T0 in enumerate(T0_vals):
                ela_data[j,k,l] = calc_ela(P0, T0, gamma, mu)
    
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
