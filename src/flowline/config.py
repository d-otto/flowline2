from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FlowlineConfig:
    """Configuration parameters for the flowline model"""
    # Physical parameters
    rho: float = 916.8  # Ice density kg/m^3
    g: float = 9.81     # Gravity m/s^2
    fd: float = 2.4e-24 # Deformation parameter Pa^-3 s^-2
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
    xlim0: Optional[float] = None        # Left limit for plots
    
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
