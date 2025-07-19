import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

def create_correlated_series(series1, series2, target_correlation, 
                           preserve_first_series=True, standardize_output=True):
    """
    Create two correlated time series using Cholesky decomposition.
    
    Parameters:
    -----------
    series1, series2 : array-like
        Input time series (should be same length)
    target_correlation : float
        Desired correlation coefficient between output series (-1 to 1)
    preserve_first_series : bool, default=True
        If True: First output series keeps the same relative pattern as series1
        If False: Both output series will be transformed 
    standardize_output : bool, default=True
        If True: Output series will have mean=0, std=1
        If False: Output series will have original scales
        
    Returns:
    --------
    y1, y2 : numpy arrays
        The two correlated time series
        
    Notes:
    ------
    - When preserve_first_series=True: y1 will have the same pattern as series1
    - When preserve_first_series=False: Both y1 and y2 are new combinations
    - Input series are automatically standardized before transformation
    """
    
    # Convert to numpy arrays and check lengths
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    if len(s1) != len(s2):
        raise ValueError("Series must have the same length")
    
    if not -1 <= target_correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    # Standardize input series (mean=0, std=1)
    s1_std = (s1 - np.mean(s1)) / np.std(s1)
    s2_std = (s2 - np.mean(s2)) / np.std(s2)
    
    # Create target covariance matrix
    target_cov = np.array([[1.0, target_correlation],
                          [target_correlation, 1.0]])
    
    # Compute Cholesky decomposition
    L = cholesky(target_cov, lower=True)
    
    # Choose transformation approach
    if preserve_first_series:
        # Method 1: Keep first series unchanged (up to standardization)
        # This makes L = [[1, 0], [ρ, √(1-ρ²)]]
        rho = target_correlation
        sqrt_term = np.sqrt(1 - rho**2)
        
        y1 = s1_std  # First series unchanged
        y2 = rho * s1_std + sqrt_term * s2_std  # Second series is combination
        
    else:
        # Method 2: Transform both series using full Cholesky matrix
        # This gives different (but equivalent) correlated series
        input_matrix = np.vstack([s1_std, s2_std])
        output_matrix = L @ input_matrix
        
        y1 = output_matrix[0, :]
        y2 = output_matrix[1, :]
    
    # Optionally restore original scales
    if not standardize_output:
        # Restore to original means and standard deviations
        y1 = y1 * np.std(s1) + np.mean(s1)
        y2 = y2 * np.std(s2) + np.mean(s2)
    
    return y1, y2


import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

def create_correlated_series(series1, series2, target_correlation, 
                           preserve_first_series=True, standardize_output=True):
    """
    Create two correlated time series using Cholesky decomposition.
    
    Parameters:
    -----------
    series1, series2 : array-like
        Input time series (should be same length)
    target_correlation : float
        Desired correlation coefficient between output series (-1 to 1)
    preserve_first_series : bool, default=True
        If True: First output series keeps the same relative pattern as series1
        If False: Both output series will be transformed 
    standardize_output : bool, default=True
        If True: Output series will have mean=0, std=1
        If False: Output series will have original scales
        
    Returns:
    --------
    y1, y2 : numpy arrays
        The two correlated time series
        
    Notes:
    ------
    - When preserve_first_series=True: y1 will have the same pattern as series1
    - When preserve_first_series=False: Both y1 and y2 are new combinations
    - Input series are automatically standardized before transformation
    """
    
    # Convert to numpy arrays and check lengths
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    if len(s1) != len(s2):
        raise ValueError("Series must have the same length")
    
    if not -1 <= target_correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    # Standardize input series (mean=0, std=1)
    s1_std = (s1 - np.mean(s1)) / np.std(s1)
    s2_std = (s2 - np.mean(s2)) / np.std(s2)
    
    # Create target covariance matrix
    target_cov = np.array([[1.0, target_correlation],
                          [target_correlation, 1.0]])
    
    # Compute Cholesky decomposition
    L = cholesky(target_cov, lower=True)
    
    # Choose transformation approach
    if preserve_first_series:
        # Method 1: Keep first series unchanged (up to standardization)
        # This makes L = [[1, 0], [ρ, √(1-ρ²)]]
        rho = target_correlation
        sqrt_term = np.sqrt(1 - rho**2)
        
        y1 = s1_std  # First series unchanged
        y2 = rho * s1_std + sqrt_term * s2_std  # Second series is combination
        
    else:
        # Method 2: Transform both series using full Cholesky matrix
        # This gives different (but equivalent) correlated series
        input_matrix = np.vstack([s1_std, s2_std])
        output_matrix = L @ input_matrix
        
        y1 = output_matrix[0, :]
        y2 = output_matrix[1, :]
    
    # Optionally restore original scales
    if not standardize_output:
        # Restore to original means and standard deviations
        y1 = y1 * np.std(s1) + np.mean(s1)
        y2 = y2 * np.std(s2) + np.mean(s2)
    
    return y1, y2


import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

def create_correlated_series(series1, series2, target_correlation, 
                           preserve_first_series=True, standardize_output=True):
    """
    Create two correlated time series using Cholesky decomposition.
    
    Parameters:
    -----------
    series1, series2 : array-like
        Input time series (should be same length)
    target_correlation : float
        Desired correlation coefficient between output series (-1 to 1)
    preserve_first_series : bool, default=True
        If True: First output series keeps the same relative pattern as series1
        If False: Both output series will be transformed 
    standardize_output : bool, default=True
        If True: Output series will have mean=0, std=1
        If False: Output series will have original scales
        
    Returns:
    --------
    y1, y2 : numpy arrays
        The two correlated time series
        
    Notes:
    ------
    - When preserve_first_series=True: y1 will have the same pattern as series1
    - When preserve_first_series=False: Both y1 and y2 are new combinations
    - Input series are automatically standardized before transformation
    """
    
    # Convert to numpy arrays and check lengths
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    if len(s1) != len(s2):
        raise ValueError("Series must have the same length")
    
    if not -1 <= target_correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    # Standardize input series (mean=0, std=1)
    s1_std = (s1 - np.mean(s1)) / np.std(s1)
    s2_std = (s2 - np.mean(s2)) / np.std(s2)
    
    # Create target covariance matrix
    target_cov = np.array([[1.0, target_correlation],
                          [target_correlation, 1.0]])
    
    # Compute Cholesky decomposition
    L = cholesky(target_cov, lower=True)
    
    # Choose transformation approach
    if preserve_first_series:
        # Method 1: Keep first series unchanged (up to standardization)
        # This makes L = [[1, 0], [ρ, √(1-ρ²)]]
        rho = target_correlation
        sqrt_term = np.sqrt(1 - rho**2)
        
        y1 = s1_std  # First series unchanged
        y2 = rho * s1_std + sqrt_term * s2_std  # Second series is combination
        
    else:
        # Method 2: Transform both series using full Cholesky matrix
        # This gives different (but equivalent) correlated series
        input_matrix = np.vstack([s1_std, s2_std])
        output_matrix = L @ input_matrix
        
        y1 = output_matrix[0, :]
        y2 = output_matrix[1, :]
    
    # Optionally restore original scales
    if not standardize_output:
        # Restore to original means and standard deviations
        y1 = y1 * np.std(s1) + np.mean(s1)
        y2 = y2 * np.std(s2) + np.mean(s2)
    
    return y1, y2

import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

def create_correlated_series(series1, series2, target_correlation, 
                           preserve_first_series=True, standardize_output=True):
    """
    Create two correlated time series using Cholesky decomposition.
    
    Parameters:
    -----------
    series1, series2 : array-like
        Input time series (should be same length)
    target_correlation : float
        Desired correlation coefficient between output series (-1 to 1)
    preserve_first_series : bool, default=True
        If True: First output series keeps the same relative pattern as series1
        If False: Both output series will be transformed 
    standardize_output : bool, default=True
        If True: Output series will have mean=0, std=1
        If False: Output series will have original scales
        
    Returns:
    --------
    y1, y2 : numpy arrays
        The two correlated time series
        
    Notes:
    ------
    - When preserve_first_series=True: y1 will have the same pattern as series1
    - When preserve_first_series=False: Both y1 and y2 are new combinations
    - Input series are automatically standardized before transformation
    """
    
    # Convert to numpy arrays and check lengths
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    if len(s1) != len(s2):
        raise ValueError("Series must have the same length")
    
    if not -1 <= target_correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    # Standardize input series (mean=0, std=1)
    s1_std = (s1 - np.mean(s1)) / np.std(s1)
    s2_std = (s2 - np.mean(s2)) / np.std(s2)
    
    # Create target covariance matrix
    target_cov = np.array([[1.0, target_correlation],
                          [target_correlation, 1.0]])
    
    # Compute Cholesky decomposition
    L = cholesky(target_cov, lower=True)
    
    # Choose transformation approach
    if preserve_first_series:
        # Method 1: Keep first series unchanged (up to standardization)
        # This makes L = [[1, 0], [ρ, √(1-ρ²)]]
        rho = target_correlation
        sqrt_term = np.sqrt(1 - rho**2)
        
        y1 = s1_std  # First series unchanged
        y2 = rho * s1_std + sqrt_term * s2_std  # Second series is combination
        
    else:
        # Method 2: Transform both series using full Cholesky matrix
        # This gives different (but equivalent) correlated series
        input_matrix = np.vstack([s1_std, s2_std])
        output_matrix = L @ input_matrix
        
        y1 = output_matrix[0, :]
        y2 = output_matrix[1, :]
    
    # Optionally restore original scales
    if not standardize_output:
        # Restore to original means and standard deviations
        y1 = y1 * np.std(s1) + np.mean(s1)
        y2 = y2 * np.std(s2) + np.mean(s2)
    
    return y1, y2


def cohstat(dof,siglev):
    #Siglev, significance level desired, should be .95 or .99. Siglev greater
    #than .95 will default to .99, lower inputs of siglev default to .95

    # this is a rough fit to the F statistic (0.01), assuming the denominator has 100 dof.
    #f=[3.94,3.09,3.98,3.51,3.21,2.99,2.82,2.69,2.59,2.5,2.07,1.89,1.80,1.74,1.65,1.60];
    #n=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,75,100];
    # this is a rough fit to the Coherency statistic (0.01), given dof as input.
    f99 = [0.99,0.684,0.602,0.536,0.482,0.438,0.401,0.342,0.264,0.215,0.175,0.147,0.112,0.075,0.057,0.045,0.023,0.002]
    f90 = [0.901,0.437,0.370,0.319,0.280,0.250,0.226,0.189,0.142,0.112,0.091,0.076,0.057,0.038,0.029,0.023,0.011,0.001]
    f95 = [0.951,0.527,0.450,0.393,0.348,0.312,0.283,0.238,0.181,0.146,0.118,0.098,0.074,0.050,0.037,0.030,0.015,0.001]
    n=[2,5,6,7,8,9,10,12,16,20,25,30,40,60,80,100,200,1000000];

    if  siglev == 0.99:
        f = f99
    elif siglev == 0.95:
        f = f95
    else:
        raise ValueError
    coh_crit = np.interp(dof,n,f)
    return coh_crit