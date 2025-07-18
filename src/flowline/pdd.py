import numpy as np
from scipy.special import erfc
import numba
from numba import jit, njit

def pdd_scipy(T_ma, T_mj, sigma, A=1.0, n_steps=365):
    """
    Calculate positive degree days using scipy's erfc function.
    
    Parameters:
    -----------
    T_ma : float
        Mean annual surface-air temperature (°C)
    T_mj : float  
        Mean July (January) surface-air temperature (°C)
    sigma : float
        Standard deviation of temperature from annual cycle (°C)
    A : float, optional
        Period length in years (default: 1.0)
    n_steps : int, optional
        Number of time steps for integration (default: 365)
        
    Returns:
    --------
    float
        Positive degree days
    """
    # Time array
    t = np.linspace(0, A, n_steps)
    dt = A / n_steps
    
    # Annual temperature cycle (sinusoidal)
    T_ac = T_ma + (T_mj - T_ma) * np.cos(2 * np.pi * t / A)
    
    # Calculate the integrand from equation (6)
    term1 = sigma / np.sqrt(2 * np.pi) * np.exp(-T_ac**2 / (2 * sigma**2))
    term2 = T_ac / 2 * erfc(-T_ac / (np.sqrt(2) * sigma))
    
    integrand = term1 + term2
    
    # Integrate using trapezoidal rule
    pdd = np.trapz(integrand, dx=dt)
    
    return pdd


@jit(nopython=True)
def erfc_approx(x):
    """
    Approximation of the complementary error function for numba compatibility.
    Uses Abramowitz and Stegun approximation (equation 7.1.26).
    
    Parameters:
    -----------
    x : float or array
        Input value(s)
        
    Returns:
    --------
    float or array
        Approximation of erfc(x)
    """
    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # Handle scalar input
    if np.isscalar(x):
        if x < 0:
            return 2.0 - erfc_approx(-x)
        
        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + p * x)
        return t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * np.exp(-x * x)
    
    # Handle array input
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < 0:
            result[i] = 2.0 - erfc_approx(-x[i])
        else:
            t = 1.0 / (1.0 + p * x[i])
            result[i] = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * np.exp(-x[i] * x[i])
    
    return result


@njit
def pdd_numba(T_ma, T_mj, sigma, A=1.0, n_steps=365):
    """
    Calculate positive degree days using numba-compatible implementation.
    
    Parameters:
    -----------
    T_ma : float
        Mean annual surface-air temperature (°C)
    T_mj : float  
        Mean July (January) surface-air temperature (°C)
    sigma : float
        Standard deviation of temperature from annual cycle (°C)
    A : float, optional
        Period length in years (default: 1.0)
    n_steps : int, optional
        Number of time steps for integration (default: 365)
        
    Returns:
    --------
    float
        Positive degree days
    """
    # Time step
    dt = A / n_steps
    
    # Initialize sum
    pdd_sum = 0.0
    
    # Integration loop
    for i in range(n_steps):
        t = i * dt
        
        # Annual temperature cycle (sinusoidal)
        T_ac = T_ma + (T_mj - T_ma) * np.cos(2 * np.pi * t / A)
        
        # Calculate the integrand from equation (6)
        term1 = sigma / np.sqrt(2 * np.pi) * np.exp(-T_ac**2 / (2 * sigma**2))
        term2 = T_ac / 2 * erfc_approx(-T_ac / (np.sqrt(2) * sigma))
        
        integrand = term1 + term2
        
        # Add to sum (rectangular rule for simplicity in numba)
        pdd_sum += integrand * dt
    
    return pdd_sum


# Example usage and comparison
if __name__ == "__main__":
    # Example parameters from the paper (south Greenland ablation zone)
    T_ma = -10.0  # Mean annual temperature (°C)
    T_mj = 5.0    # Mean July temperature (°C)
    sigma = 5.0   # Standard deviation (°C)
    
    # Calculate PDD using both methods
    pdd_scipy_result = pdd_scipy(T_ma, T_mj, sigma)
    pdd_numba_result = pdd_numba(T_ma, T_mj, sigma)
    
    print(f"PDD (scipy): {pdd_scipy_result:.2f} °C·days")
    print(f"PDD (numba): {pdd_numba_result:.2f} °C·days")
    print(f"Difference: {abs(pdd_scipy_result - pdd_numba_result):.4f} °C·days")
    
    # Timing comparison
    import time
    
    # Time scipy version
    start = time.time()
    for _ in range(1000):
        pdd_scipy(T_ma, T_mj, sigma)
    scipy_time = time.time() - start
    
    # Time numba version (first call includes compilation)
    start = time.time()
    for _ in range(1000):
        pdd_numba(T_ma, T_mj, sigma)
    numba_time = time.time() - start
    
    print(f"\nTiming (1000 iterations):")
    print(f"Scipy version: {scipy_time:.4f} seconds")
    print(f"Numba version: {numba_time:.4f} seconds")
    print(f"Speedup: {scipy_time/numba_time:.1f}x")