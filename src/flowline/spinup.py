"""
FlowlineSpinup class for generating steady-state profiles and applying perturbations.

This module provides functionality to:
1. Generate steady-state glacier profiles for specific parameter sets
2. Apply target matching to achieve comparable glacier states
3. Apply perturbations for response testing
4. Integrate cleanly with the FlowlineSweep architecture
"""

from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import math
import logging

import numpy as np
from scipy.optimize import minimize
from flowline.entrypoints import run_spinup_simulation
from flowline.utils import objects_equal, object_hash
from tqdm import tqdm

# Set up logger
logger = logging.getLogger(__name__)

def setup_optimization_logging(output_dir, run_id):
    """
    Set up file-based logging for optimization tracking.
    
    Creates separate log files for each run_id to avoid conflicts in parallel execution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create a unique logger for this run_id
    logger_name = f"optimization_{run_id}"
    opt_logger = logging.getLogger(logger_name)
    opt_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    for handler in opt_logger.handlers[:]:
        opt_logger.removeHandler(handler)
    
    # Set up file handler for this specific run
    log_file = output_dir / f"optimization_{run_id}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter that includes timestamp and run_id
    formatter = logging.Formatter(
        f'%(asctime)s - {run_id} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    opt_logger.addHandler(file_handler)
    opt_logger.propagate = False  # Prevent duplicate messages in root logger
    
    return opt_logger


# =============================================================================
# TQDM-SAFE PRINTING UTILITIES
# =============================================================================

def safe_print(message, use_tqdm=True):
    """Print message in a way that's compatible with tqdm progress bars."""
    if use_tqdm:
        try:
            tqdm.write(message)
        except Exception:
            print(message)
    else:
        print(message)


# =============================================================================
# MODULAR COST FUNCTION SYSTEM
# =============================================================================

class CostFunction(ABC):
    """
    Abstract base class for cost functions in target matching optimization.
    
    Cost functions evaluate how well the current glacier state matches the target
    state. They should return a scalar value where 0 means perfect match and
    higher values indicate larger deviations from the target.
    """
    
    @abstractmethod
    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        """
        Calculate cost between current state and targets.
        
        Parameters
        ----------
        model_state : dict
            Current glacier state containing:
            - 'edge': glacier terminus index
            - 'delx': grid spacing
            - 'h': ice thickness array
            - 'volume': total glacier volume
            - 'area': glacier area
            - Any other relevant state variables
        targets : dict
            Target values for optimization containing keys like:
            - 'target_length': target glacier length (m)
            - 'target_avg_thickness': target average thickness (m)
            - 'target_volume': target glacier volume (m³)
            - Any other relevant targets
            
        Returns
        -------
        float
            Cost value (0 = perfect match, higher = worse match)
        """
        pass

    def initial_guess(self, geometry: Any, forcing: Any, targets: Dict[str, Any]) -> Optional[float]:
        """
        Optional method to provide initial parameter guess for optimization.
        
        Parameters
        ----------
        geometry : FlowlineGeometry
            Geometry object for the simulation
        forcing : MassBalanceForcing
            Forcing object for the simulation
        targets : dict
            Target values for optimization
            
        Returns
        -------
        float or None
            Initial guess for the parameter to optimize, or None if no guess available
        """
        return None
    
    def plot_cost_curve(self, targets: Dict[str, Any], domain_length: float = 16000, 
                       delx: float = 50, output_file: Optional[str] = None):
        """
        Plot the cost function curve across the range of possible glacier lengths.
        
        Parameters
        ----------
        targets : dict
            Target values for optimization (e.g., {'target_length': 8000})
        domain_length : float, optional
            Total domain length in meters (default: 16000m = 16km)
        delx : float, optional
            Grid spacing in meters (default: 50m)
        output_file : str, optional
            Path to save the plot. If None, displays the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib and numpy are required for plotting cost curves")
            return
        
        # Create range of possible lengths
        max_grid_points = int(domain_length / delx)
        edge_indices = np.arange(0, max_grid_points + 1)
        lengths = edge_indices * delx
        
        # Calculate costs for each length
        costs = []
        for i, length in enumerate(lengths):
            # Create a mock model state for cost calculation
            mock_state = {
                'edge': edge_indices[i],
                'delx': delx,
                'h': np.zeros(max_grid_points),  # Mock ice thickness array
            }
            
            try:
                cost = self(mock_state, targets)
                # Cap very large costs for better visualization
                cost = min(cost, 1e6)
                costs.append(cost)
            except Exception as e:
                # Handle any errors in cost calculation
                costs.append(np.nan)
        
        costs = np.array(costs)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the cost curve
        valid_mask = np.isfinite(costs)
        ax.plot(lengths[valid_mask] / 1000, costs[valid_mask], 'b-', linewidth=2, label='Cost Function')
        
        # Mark the target length if it exists
        if 'target_length' in targets:
            target_length = targets['target_length']
            ax.axvline(x=target_length / 1000, color='red', linestyle='--', linewidth=2, 
                      label=f'Target Length ({target_length/1000:.1f} km)')
            
            # Mark the cost at target length
            target_idx = int(target_length / delx)
            if target_idx < len(costs) and np.isfinite(costs[target_idx]):
                ax.plot(target_length / 1000, costs[target_idx], 'ro', markersize=8, 
                       label=f'Target Cost ({costs[target_idx]:.3f})')
        
        # Formatting
        ax.set_xlabel('Glacier Length (km)')
        ax.set_ylabel('Cost Function Value')
        ax.set_title(f'Cost Function: {self.__class__.__name__}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Use log scale for y-axis if costs span many orders of magnitude
        if np.any(valid_mask):
            cost_range = np.max(costs[valid_mask]) / max(1e-6, np.min(costs[valid_mask]))
            if cost_range > 100:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Cost curve plot saved to: {output_file}")
        else:
            plt.show()
        
        return fig, ax


class LengthOnlyCost(CostFunction):
    """
    Symmetric exponential cost function for glacier length optimization.
    
    Creates a symmetric cost function where lengths equidistant from the target
    (on either side) have equal cost. Uses exponential scaling to provide
    stronger penalties for larger deviations from target length.
    
    For a domain of length D and target length T:
    - Scale factor = min(T, D-T) ensures equal cost at domain extremes
    - cost = exp(|current_length - target_length| / scale_factor) - 1
    - cost(T) = 0 (perfect match)
    - cost(0) = cost(D) when T is at domain center
    """
    def __init__(self):
        self._debug_logger = None

    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        edge_idx = model_state['edge']
        current_length = edge_idx * model_state['delx']
        target_length = targets['target_length']
        
        # Get domain extent for symmetric scaling
        # Estimate domain size from delx and model dimensions
        delx = model_state['delx']
        domain_length = len(model_state['h']) * delx
        
        # Base cost
        distance_from_target = abs(current_length - target_length)
        cost = distance_from_target
        
        # Add a continuous component based on volume or thickness
        # This helps the optimizer distinguish between glaciers of same length
        # But only if we're pretty far off
        if (edge_idx > 0) & (distance_from_target > domain_length//4):
            h = model_state['h']
            min_thickness = np.min(h[:edge_idx])
            # Penalty for being too thick (encourages warmer temps)
            thickness_penalty = min_thickness * 100  # Tune this weight
            cost += thickness_penalty
        
        # Debug logging - check if we have access to logger
        if self._debug_logger:
            self._debug_logger.debug(f"COST_DEBUG: edge_idx={edge_idx}, current_length={current_length:.1f}m")
            self._debug_logger.debug(f"COST_DEBUG: target_length={target_length:.1f}m, domain_length={domain_length:.1f}m")
        
        return cost

    def initial_guess(self, geometry: Any, forcing: Any, targets: Dict[str, Any]) -> Optional[float]:
        """Calculate initial guess for glacier length based on ELA and geometry."""
        try:
            # Import here to avoid circular imports
            from flowline.diagnostics import calc_ela
            
            # Calculate ELA from mass balance parameters - let it fail if attributes missing
            ela = calc_ela(forcing.P0, forcing.T0, forcing.gamma, forcing.mu)
            
            # Find where bed elevation equals ELA
            # Use the high-resolution geometry grid for accuracy
            x_gr = geometry.x_gr
            zb_gr = geometry.zb_gr
            
            # Find the index where bed elevation is closest to ELA
            ela_idx_hr = np.argmin(np.abs(zb_gr - ela))
            
            # Get the distance from origin to ELA
            ela_distance = x_gr[ela_idx_hr]
            
            # Check if we have actual width information (all non-zero widths)
            if not np.all(geometry.w_geom == 0):
                # Use area-based calculation
                
                # Calculate area above ELA (60% of total area assumption)
                # First estimate total area up to ELA
                area_to_ela = np.sum(geometry.w_geom[:ela_idx_hr]) * (x_gr[1] - x_gr[0])
                
                # Total area should be such that 60% is above ELA
                # So: area_above_ela = 0.6 * total_area
                # And: area_above_ela ≈ area_to_ela (rough approximation)
                # Therefore: total_area ≈ area_to_ela / 0.6
                estimated_total_area = area_to_ela / 0.6
                
                # Find the length that gives this total area
                # Work backwards from area to find terminus position
                cumulative_area = 0.0
                dx = x_gr[1] - x_gr[0]
                
                for i in range(len(x_gr)):
                    cumulative_area += geometry.w_geom[i] * dx
                    if cumulative_area >= estimated_total_area:
                        return x_gr[i]
                        
            else:
                # Use length-based calculation (no width info)
                # Assume 60% of glacier length is above ELA
                # So: length_above_ela = 0.6 * total_length
                # And: length_above_ela ≈ ela_distance
                # Therefore: total_length ≈ ela_distance / 0.6
                return ela_distance / 0.6
                
        except Exception:
            # If initial guess fails, return None
            return None


class AreaOnlyCost(CostFunction):
    """Cost function for glacier area optimization.

    Minimizes the absolute difference between current glacier area and a target
    area. Area is computed as sum(w * delx) over all ice-covered grid cells.
    """

    def __init__(self):
        self._debug_logger = None

    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        current_area = model_state['area']  # m²
        target_area = targets['target_area']  # m²

        cost = abs(current_area - target_area)

        if self._debug_logger:
            self._debug_logger.debug(
                f"COST_DEBUG: current_area={current_area:.1f} m², "
                f"target_area={target_area:.1f} m², cost={cost:.1f}"
            )

        return cost

    def initial_guess(self, geometry: Any, forcing: Any, targets: Dict[str, Any]) -> Optional[float]:
        return None


class VolumeOnlyCost(CostFunction):
    """Cost function for glacier volume optimization.

    Minimizes the absolute difference between current glacier volume and a
    target volume. Volume is computed as sum(h * w * delx) over all grid cells.
    """

    def __init__(self):
        self._debug_logger = None

    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        current_volume = model_state['volume']  # m³
        target_volume = targets['target_volume']  # m³

        cost = abs(current_volume - target_volume)

        if self._debug_logger:
            self._debug_logger.debug(
                f"COST_DEBUG: current_volume={current_volume:.4e} m³, "
                f"target_volume={target_volume:.4e} m³, cost={cost:.4e}"
            )

        return cost

    def initial_guess(self, geometry: Any, forcing: Any, targets: Dict[str, Any]) -> Optional[float]:
        return None


class LengthAndAverageThicknessCost(CostFunction):
    """Cost function that considers both glacier length and average thickness."""
    
    def __init__(self, length_weight=1.0, thickness_weight=1.0):
        """
        Initialize with optional weighting.
        
        Parameters
        ----------
        length_weight : float
            Weight for length component of cost
        thickness_weight : float
            Weight for thickness component of cost
        """
        self.length_weight = length_weight
        self.thickness_weight = thickness_weight
    
    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        # Length component
        current_length = model_state['edge'] * model_state['delx']
        target_length = targets['target_length']
        length_cost = abs(current_length - target_length)
        
        # Average thickness component
        h = model_state['h']
        active_ice_mask = h > 0
        if np.any(active_ice_mask):
            avg_thickness = np.mean(h[active_ice_mask])
        else:
            avg_thickness = 0.0
        
        target_avg_thickness = targets['target_avg_thickness']
        thickness_cost = abs(avg_thickness - target_avg_thickness)
        
        return (self.length_weight * length_cost + 
                self.thickness_weight * thickness_cost)


# =============================================================================
# MODULAR STEADY-STATE DETECTION SYSTEM
# =============================================================================

class SteadyStateDetector(ABC):
    """
    Abstract base class for steady-state detection methods.
    
    Steady-state detectors determine when a glacier simulation has reached
    equilibrium and is ready for target matching evaluation.
    """
    
    @abstractmethod
    def __call__(self, model_state: Dict[str, Any]) -> bool:
        """
        Check if glacier has reached steady state.
        
        Parameters
        ----------
        model_state : dict
            Current glacier state containing time series data like:
            - 'volume_history': time series of glacier volume
            - 'length_history': time series of glacier length
            - 'time_history': time series of simulation time
            - Any other relevant time series
            
        Returns
        -------
        bool
            True if steady state has been reached, False otherwise
        """
        pass


class VolumeChangeRateDetector(SteadyStateDetector):
    """Steady-state detector based on volume change rate (dV/dt)."""
    
    def __init__(self, threshold=1e3, window_size=10, min_time=100):
        """
        Initialize detector parameters.
        
        Parameters
        ----------
        threshold : float
            Maximum allowable |dV/dt| for steady state (default units: relative/year)
        window_size : int
            Number of recent timesteps to analyze
        min_time : float
            Minimum simulation time before checking steady state (years)
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_time = min_time
    
    def __call__(self, model_state: Dict[str, Any]) -> bool:
        time_history = model_state.get('time_history', [])
        volume_history = model_state.get('volume_history', [])
        
        # Need minimum time and data points
        if (len(time_history) < self.window_size or 
            len(time_history) == 0 or 
            time_history[-1] < self.min_time):
            return False
        
        # Calculate volume change rate over recent window
        recent_times = np.array(time_history[-self.window_size:])
        recent_volumes = np.array(volume_history[-self.window_size:])
        
        # Skip if we don't have enough valid data
        if len(recent_times) < 2:
            return False
        
        # Calculate dV/dt using finite differences
        dt = np.diff(recent_times)
        dV = np.diff(recent_volumes)
        
        # Avoid division by zero
        valid_mask = dt > 0
        if not np.any(valid_mask):
            return False
        
        dV_dt = dV[valid_mask] / dt[valid_mask]
        
        # Check if average rate of change is below threshold
        mean_dV_dt = np.mean(np.abs(dV_dt))
        
        # # Debug output
        # if len(time_history) % 50 == 0:  # Print every 50 steps
        #     safe_print(f"  Steady state check: t={time_history[-1]:.1f}, mean_dV_dt={mean_dV_dt:.2e}, threshold={self.threshold:.2e}")
        
        return bool(mean_dV_dt < self.threshold)


# =============================================================================
# RESPONSE TIME AND INITIAL GUESS FUNCTIONS
# =============================================================================


class FlowlineSpinup:
    """
    Generates steady-state profiles for a single parameter set.
    
    Follows the clean 3-object architecture: takes FlowlineConfig, FlowlineGeometry, 
    and MassBalanceForcing objects. Generates a steady-state profile and optionally applies
    target matching for comparable experiments. Experimental perturbations are handled 
    separately by FlowlineSweep to maintain clean separation of concerns.
    """
    
    def __init__(self, config: Any, geometry: Any, forcing: Any, target_matching: Optional[Dict[str, Any]] = None):
        """
        Initialize FlowlineSpinup for a single parameter set.
        
        Parameters
        ----------
        config : FlowlineConfig
            Configuration for the spinup run (one parameter set)
        geometry : FlowlineGeometry  
            Geometry for the spinup run (one parameter set)
        forcing : MassBalanceForcing
            Forcing for the spinup run (one parameter set)
        target_matching : dict, optional
            Configuration for iterative target matching optimization:
            {
                'targets': {
                    'target_length': 8000,                      # Target glacier length (m)
                    'target_avg_thickness': 100                 # Target average thickness (m) [optional]
                },
                'adjustment_parameter': 'T0',                   # Parameter name to optimize
                'cost_function': 'length_only',                 # Cost function type or custom function
                'steady_state_detector': 'volume_change_rate',  # Steady-state detector type or custom
                'tolerance': 100,                               # Acceptable cost for convergence
                'parameter_bounds': (5.0, 12.0),               # Search bounds for optimization
                'max_iterations': 10,                           # Maximum optimization iterations
                'max_simulation_time': 1000                     # Maximum simulation time for optimization
            }
        """
        self.config = deepcopy(config)
        self.geometry = deepcopy(geometry)
        self.forcing = deepcopy(forcing)
        self.target_matching = target_matching
        
        # Ensure spinup timeframe is consistent between config and forcing
        if hasattr(self.forcing, 'tf'):
            self.forcing.tf = self.config.tf
    
    @property
    def cost_function(self):
        """Get cost function, instantiating if needed.
        
        Accesses target_matching['cost_function'] and instantiates it if it's a class.
        Will fail with KeyError if target_matching is missing or doesn't contain 'cost_function'.
        This is intentional - no silent defaults.
        """
        if self.target_matching:
            spec = self.target_matching['cost_function']
            return spec() if isinstance(spec, type) else spec
        return None
    
    @property 
    def steady_state_detector(self):
        """Get steady-state detector, instantiating if needed.
        
        Accesses target_matching['steady_state_detector'] and instantiates it if it's a class.
        Will fail with KeyError if target_matching is missing or doesn't contain 'steady_state_detector'.
        This is intentional - no silent defaults.
        """
        if self.target_matching:
            spec = self.target_matching['steady_state_detector']
            return spec() if isinstance(spec, type) else spec
        return None
    
    
    def generate_profile(self, output_dir, run_id, no_progress=False):
        """
        Generate steady-state profile with optional target matching optimization.
        
        Parameters
        ----------
        output_dir : Path or str
            Directory to save spinup results
        run_id : str
            Unique identifier for this run
        no_progress : bool, optional
            Disable progress bars
            
        Returns
        -------
        tuple
            (profile_path, optimized_parameters) where optimized_parameters 
            is a dict of parameter names to optimized values
        """
        output_dir = Path(output_dir)
        spinup_id = f"spinup_{run_id}"
        
        # Set up logging for this specific run_id
        self.opt_logger = setup_optimization_logging(output_dir, run_id)
        self.opt_logger.info(f"Starting target matching optimization for {run_id}")
        
        optimized_parameters = {}
        if self.target_matching:
            self.opt_logger.info("Using optimization-based target matching")
            # Use optimization-based target matching
            optimized_param = self._optimize_target_matching(output_dir, spinup_id, no_progress)
            # Get parameter name for backward compatibility
            if 'adjustment_parameter' in self.target_matching:
                param_name = self.target_matching['adjustment_parameter']
                self.opt_logger.info(f"Target matching optimization completed. Optimal {param_name} = {optimized_param[0]:.3f}")
                safe_print(f"Target matching optimization completed. Optimal {param_name} = {optimized_param[0]:.3f}")
                optimized_parameters[param_name] = optimized_param[0]
            else:
                param_names = self.target_matching['adjustment_parameters']
                if len(param_names) == 1:
                    param_name = param_names[0]
                    self.opt_logger.info(f"Target matching optimization completed. Optimal {param_name} = {optimized_param[0]:.3f}")
                    safe_print(f"Target matching optimization completed. Optimal {param_name} = {optimized_param[0]:.3f}")
                    optimized_parameters[param_name] = optimized_param[0]
                else:
                    for i, (param, value) in enumerate(zip(param_names, optimized_param)):
                        self.opt_logger.info(f"Target matching optimization completed. Optimal {param} = {value:.3f}")
                        safe_print(f"Target matching optimization completed. Optimal {param} = {value:.3f}")
                        optimized_parameters[param] = value
        else:
            self.opt_logger.info("No target matching configured, running direct spinup")
        
        # Run final spinup with optimized parameters
        self.opt_logger.info(f"Running final spinup simulation for {spinup_id}")
        profile_path = self._run_steady_state_spinup(output_dir, spinup_id, no_progress)
        
        self.opt_logger.info(f"Spinup completed. Profile saved to: {profile_path}")
        return profile_path, optimized_parameters
    
    def _optimize_target_matching(self, output_dir, spinup_id, no_progress):
        """
        Optimize parameters using scipy.minimize to achieve target matching.
        
        Parameters
        ----------
        output_dir : Path
            Directory for optimization runs
        spinup_id : str
            Base identifier for optimization runs
        no_progress : bool
            Disable progress bars
            
        Returns
        -------
        list
            Optimal parameter values
        """
        # Support both old and new configuration formats
        if not self.target_matching:
            return []
        
        targets = self.target_matching['targets']
        optimization_options = self.target_matching.get("optimization_options", {})
        max_iterations = optimization_options['maxiter']
        
        # Support both old and new configuration formats
        if 'adjustment_parameter' in self.target_matching:
            # Legacy single parameter format
            adjustment_parameters = [self.target_matching['adjustment_parameter']]
            parameter_bounds = [self.target_matching['bounds']]
        else:
            # New multi-parameter format
            adjustment_parameters = self.target_matching['adjustment_parameters']
            parameter_bounds = self.target_matching['bounds']
        
        # Log optimization configuration
        self.opt_logger.info(f"Starting minimization")
        self.opt_logger.info(f"Parameters to optimize: {adjustment_parameters}")
        self.opt_logger.info(f"Parameter bounds: {parameter_bounds}")
        self.opt_logger.info(f"Targets: {targets}")
        self.opt_logger.info(f"Max iterations: {max_iterations}")
        self.opt_logger.info(f"Optimization options: {optimization_options}")
        
        # Log comprehensive model setup before optimization starts
        self.opt_logger.info("=" * 60)
        self.opt_logger.info("BASELINE MODEL SETUP FOR TARGET MATCHING OPTIMIZATION")
        self.opt_logger.info("=" * 60)
        try:
            from flowline.diagnostics import log_model_setup
            # Create a StringIO to capture the setup log and redirect to our logger
            from io import StringIO
            import sys
            
            # Capture log_model_setup output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                log_model_setup(self)
                setup_log = captured_output.getvalue()
                
                # Log each line with our optimization logger
                for line in setup_log.split('\n'):
                    if line.strip():  # Skip empty lines
                        self.opt_logger.info(line)
                        
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            self.opt_logger.warning(f"Failed to log model setup: {e}")
        
        self.opt_logger.info("=" * 60)
        
        # Initialize optimization history tracking
        self._optimization_history = {'params': [], 'costs': [], 'lengths': [], 'states': []}
        
        def objective_function(param_values):
            """
            Objective function for optimization.
            
            Runs simulation with given parameter values and returns cost.
            """
            # # Check for non-finite parameters to prevent crashes
            # if not np.all(np.isfinite(param_values)):
            #     self.opt_logger.warning(f"Received non-finite parameters: {param_values}. Returning high cost.")
            #     return 1e12

            # Robust parameter extraction and type handling
            if len(adjustment_parameters) == 1:
                # Handle various ways SHGO can pass single parameters
                if np.isscalar(param_values):
                    param_value = float(param_values)
                elif hasattr(param_values, '__len__') and len(param_values) == 1:
                    param_value = float(param_values[0])
                else:
                    # Fallback for other iterable types
                    try:
                        param_value = float(next(iter(param_values)))
                    except (TypeError, StopIteration):
                        param_value = float(param_values)
                
                param_dict = {adjustment_parameters[0]: param_value}
                param_str = f"{adjustment_parameters[0]} = {param_value:.3f}"
                opt_id = f"{spinup_id}_opt_{param_value:.3f}"
            else:
                # Multi-parameter case - ensure we have a sequence
                if not hasattr(param_values, '__len__'):
                    param_values = [param_values]
                
                param_dict = {}
                for i, param in enumerate(adjustment_parameters):
                    if i < len(param_values):
                        param_dict[param] = float(param_values[i])
                    else:
                        raise ValueError(f"Missing parameter value for {param}")
                
                param_str = ", ".join([f"{param} = {value:.3f}" for param, value in param_dict.items()])
                param_hash = "_".join([f"{value:.3f}" for value in param_values])
                opt_id = f"{spinup_id}_opt_{param_hash}"
            
            # Create fresh forcing object for this iteration to avoid state contamination
            optimization_forcing = deepcopy(self.forcing)
            
            # Set the parameter values on the fresh forcing object
            for param, value in param_dict.items():
                if not hasattr(optimization_forcing, param):
                    raise AttributeError(f"Forcing object has no attribute '{param}'")
                
                old_value = getattr(optimization_forcing, param, None)
                setattr(optimization_forcing, param, value)
                new_value = getattr(optimization_forcing, param)
                
                self.opt_logger.debug(f"PARAM_SET: {param} = {value:.6f} (type: {type(value)}, old: {old_value}, new: {new_value})")
                
                # Verify the parameter was set correctly
                if abs(new_value - value) > 1e-10:
                    self.opt_logger.error(f"Parameter setting failed: expected {value}, got {new_value}")
            
            # Temporarily replace self.forcing for this iteration
            original_forcing = self.forcing
            self.forcing = optimization_forcing
            
            # Log forcing object state for debugging
            self._log_forcing_state(optimization_forcing, "OPTIMIZATION_FORCING")
            
            # Connect debug logger to cost function for this iteration
            if hasattr(self, 'opt_logger'):
                if self.cost_function:
                    self.cost_function._debug_logger = self.opt_logger
            
            # Log current iteration setup (brief version for each iteration)
            self.opt_logger.info("-" * 40)
            self.opt_logger.info(f"OPTIMIZATION ITERATION SETUP")
            self.opt_logger.info(f"Testing parameters: {param_str}")
            
            # Log key forcing parameters for this iteration
            if hasattr(optimization_forcing, 'T0'):
                self.opt_logger.info(f"Current forcing T0: {optimization_forcing.T0:.3f}°C")
            if hasattr(optimization_forcing, 'P0'):
                self.opt_logger.info(f"Current forcing P0: {optimization_forcing.P0:.3f} m w.e./yr")
            
            # Calculate ELA for this temperature to verify climate forcing is working
            try:
                from flowline.diagnostics import calc_ela
                ela = calc_ela(self.forcing.P0, self.forcing.T0, self.forcing.gamma, self.forcing.mu)
                self.opt_logger.info(f"Estimated ELA for this iteration: {ela:.1f}m")
                self.opt_logger.info(f"OPTIMIZATION ITERATION: Trying {param_str}, ELA = {ela:.1f}m")
                print(f"Optimization trying: {param_str}, ELA = {ela:.1f}m")
            except Exception as e:
                self.opt_logger.info(f"OPTIMIZATION ITERATION: Trying {param_str} (ELA calc failed: {e})")
                print(f"Optimization trying: {param_str} (ELA calc failed: {e})")
                
            self.opt_logger.info("-" * 40)
            
            # Run simulation with inline optimization monitoring
            try:
                cost = self._run_simulation_with_monitoring(output_dir, opt_id, no_progress)
            except RuntimeError as e:
                self.opt_logger.warning(f"Simulation failed for params {param_str}: {e}")
                return 1e8  # High cost for failed simulation
            
            self.opt_logger.info(f"OPTIMIZATION RESULT: {param_str}, cost = {cost:.1f}")
            print(f"Optimization result: {param_str}, cost = {cost:.1f}")
            
            # Add diagnostic info about the final state and store optimization history
            if hasattr(self, '_last_optimization_state'):
                state = self._last_optimization_state
                final_length = state["final_length"]
                final_edge_idx = state["final_edge_idx"]
                steady_state_time = state["steady_state_time"]
                final_volume = state.get("final_volume", 0)
                
                self.opt_logger.info(f"SIMULATION STATE: Final length: {final_length:.1f}m (edge_idx={final_edge_idx}), steady state at: {steady_state_time}, volume: {final_volume:.1f}")
                safe_print(f"  -> Final length: {final_length:.1f}m (edge_idx={final_edge_idx}), steady state at: {steady_state_time}")
                
                # Store in optimization history
                self._optimization_history['params'].append(param_values if len(adjustment_parameters) > 1 else param_values)
                self._optimization_history['costs'].append(cost)
                self._optimization_history['lengths'].append(final_length)
                self._optimization_history['states'].append(state)
            else:
                self.opt_logger.warning("SIMULATION WARNING: No optimization state available - simulation may not have reached steady state")
                safe_print("  -> No optimization state available")
                self._optimization_history['params'].append(param_values if len(adjustment_parameters) > 1 else param_values)
                self._optimization_history['costs'].append(cost)
                self._optimization_history['lengths'].append(0)
                self._optimization_history['states'].append({})
            
            # Restore original forcing object to prevent state contamination
            self.forcing = original_forcing
            
            return cost
        
        # Validate parameter bounds
        if not parameter_bounds or len(parameter_bounds) != len(adjustment_parameters):
            raise ValueError(f"Parameter bounds must be provided for all {len(adjustment_parameters)} parameters")
        
        for i, (param, bounds_tuple) in enumerate(zip(adjustment_parameters, parameter_bounds)):
            if not isinstance(bounds_tuple, (list, tuple)) or len(bounds_tuple) != 2:
                raise ValueError(f"Bounds for parameter '{param}' must be a tuple/list of (min, max)")
            if bounds_tuple[0] >= bounds_tuple[1]:
                raise ValueError(f"Lower bound must be less than upper bound for parameter '{param}': {bounds_tuple}")
        
        # Log final configuration
        self.opt_logger.info(f"MINIMIZE CONFIG: {optimization_options}")
        
        # Provide initial guess (x0) - use the center of the bounds
        x0 = [(b[0] + b[1]) / 2 for b in parameter_bounds]
        
        # Run optimization with scipy.minimize
        result = minimize(
            objective_function,
            x0=x0,
            method='Nelder-Mead',  # A good choice for bound-constrained problems
            options=optimization_options
        )
        
        # Log optimization completion
        self.opt_logger.info(f"Minimization completed: success={result.success}")
        self.opt_logger.info(f"Final result: {result.x}, function value: {result.fun}")
        self.opt_logger.info(f"Number of function evaluations: {result.nfev}")
        
        if not result.success:
            self.opt_logger.warning(f"Optimization did not converge. Using best result: {result.x}")
            safe_print(f"Warning: Optimization did not converge. Using best result: {result.x}")
        
        # Set the optimal parameter values
        optimal_params = result.x.tolist()
        for param, value in zip(adjustment_parameters, optimal_params):
            # Ensure value is a scalar for setting on forcing object
            scalar_value = float(value)
            setattr(self.forcing, param, scalar_value)
            self.opt_logger.info(f"FINAL PARAMETER: Set {param}={scalar_value:.3f} on forcing object")
        
        # Log optimization history summary
        if hasattr(self, '_optimization_history'):
            costs = self._optimization_history['costs']
            params = self._optimization_history['params']
            lengths = self._optimization_history['lengths']
            
            finite_mask = np.isfinite(costs)
            finite_costs = np.array(costs)[finite_mask]
            finite_params = np.array(params)[finite_mask]
            finite_lengths = np.array(lengths)[finite_mask]

            self.opt_logger.info(f"OPTIMIZATION SUMMARY: {len(costs)} total evaluations, {len(finite_costs)} reached steady state")
            if len(finite_costs) > 0:
                best_idx = np.argmin(finite_costs)
                best_cost = finite_costs[best_idx]
                best_param = finite_params[best_idx]
                best_length = finite_lengths[best_idx]
                self.opt_logger.info(f"Best result: cost={best_cost:.3f}, param={best_param}, length={best_length:.1f}m")
                self.opt_logger.info(f"Cost range: {np.min(finite_costs):.3f} to {np.max(finite_costs):.3f}")
                self.opt_logger.info(f"Param range: {np.min(finite_params):.3f} to {np.max(finite_params):.3f}")
                self.opt_logger.info(f"Length range: {np.min(finite_lengths):.1f}m to {np.max(finite_lengths):.1f}m")
        
        # Create a simple visualization of the optimization progress
        try:
            self._create_optimization_plot(output_dir, spinup_id)
            self.opt_logger.info(f"Optimization plot saved successfully")
        except Exception as e:
            self.opt_logger.error(f"Could not create optimization plot: {e}")
            print(f"Warning: Could not create optimization plot: {e}")
        
        return optimal_params
    
    def _create_optimization_plot(self, output_dir=None, spinup_id=None):
        """Create a visualization of the optimization progress."""
        if not hasattr(self, '_optimization_history') or len(self._optimization_history['params']) == 0:
            print("No optimization history available for plotting")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Use subplot_mosaic for better layout control
            fig = plt.figure(figsize=(15, 10))
            mosaic = np.array([
                ['step_temp', 'temp_length'],
                ['step_cost', 'temp_cost']
            ])
            axes = fig.subplot_mosaic(mosaic)
            
            params = np.array(self._optimization_history['params'])
            costs = np.array(self._optimization_history['costs'])
            lengths = np.array(self._optimization_history['lengths'])
            steps = np.arange(len(params))
            
            # Filter out infinite costs for some plots
            finite_mask = np.isfinite(costs)
            
            # Handle both old and new configuration formats for parameter names
            if self.target_matching:
                if 'adjustment_parameter' in self.target_matching:
                    param_names = [self.target_matching['adjustment_parameter']]
                else:
                    param_names = self.target_matching['adjustment_parameters']
                target_length = self.target_matching['targets']['target_length']
            else:
                param_names = ['unknown']
                target_length = 8000
            
            # Handle single vs multi-parameter cases
            is_multi_param = len(param_names) > 1
            if is_multi_param:
                # For multi-parameter, params is a 2D array
                params = np.array(params).reshape(len(params), -1) if params.ndim == 1 else params
                param_name = f"{str(param_names[0])} (primary)"  # Ensure string for plotting
                primary_params = params[:, 0] if params.ndim == 2 else params
            else:
                # For single parameter, ensure params is 1D
                primary_params = params.flatten() if hasattr(params, 'flatten') else params
                param_name = str(param_names[0])  # Ensure string for plotting
            
            # Plot 1: Optimization Step vs Parameter (shows path optimizer takes)
            ax1 = axes['step_temp']
            ax1.plot(steps, primary_params, 'b-o', linewidth=2, markersize=6, label=f'{param_name} Path')
            ax1.set_xlabel('Optimization Step')
            ax1.set_ylabel(f'{param_name}')
            ax1.set_title(f'Optimization Path: Step vs {param_name}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Parameter vs Length (relationship)
            ax2 = axes['temp_length']
            scatter = ax2.scatter(primary_params, lengths, c=steps, cmap='viridis', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax2.plot(primary_params, lengths, 'k-', alpha=0.3, linewidth=1)
            ax2.axhline(y=target_length, color='r', linestyle='--', linewidth=2, label=f'Target ({target_length}m)')
            ax2.set_xlabel(f'{param_name}')
            ax2.set_ylabel('Glacier Length (m)')
            ax2.set_title(f'{param_name} vs Length Relationship')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add colorbar for step progression
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Optimization Step')
            
            # Plot 3: Optimization Step vs Cost (shows convergence)
            ax3 = axes['step_cost']
            
            # Separate finite and infinite costs for better visualization
            infinite_mask = ~finite_mask
            
            # Plot finite costs on main scale
            if np.any(finite_mask):
                finite_steps = steps[finite_mask]
                finite_costs = costs[finite_mask]
                ax3.plot(finite_steps, finite_costs, 'go-', linewidth=2, markersize=8, label='Finite Cost', zorder=3)
            
            # Plot infinite costs as special markers at the top
            if np.any(infinite_mask):
                infinite_steps = steps[infinite_mask]
                # Use a high value for plotting infinite costs
                max_finite = np.max(costs[finite_mask]) if np.any(finite_mask) else 1000
                plot_height = max_finite * 1.2 if max_finite > 0 else 1000
                ax3.scatter(infinite_steps, [plot_height] * len(infinite_steps), 
                           marker='^', s=100, color='red', edgecolor='darkred', 
                           linewidth=2, label='Infinite Cost', zorder=4)
                
                # Add text annotations for infinite costs
                for step in infinite_steps:
                    ax3.annotate('∞', (step, plot_height), xytext=(0, 10), 
                               textcoords='offset points', ha='center', va='bottom',
                               fontsize=12, color='darkred', weight='bold')
            
            ax3.set_xlabel('Optimization Step')
            ax3.set_ylabel('Cost Function Value')
            ax3.set_title('Optimization Convergence: Step vs Cost')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Set y-axis to show both finite and infinite regions
            if np.any(finite_mask) and np.any(infinite_mask):
                ax3.set_ylim(bottom=np.min(costs[finite_mask]) * 0.9 if np.any(finite_mask) else 0)
            
            # Plot 4: Parameter vs Cost (cost landscape)
            ax4 = axes['temp_cost']
            
            # Plot finite costs with color-coded steps
            if np.any(finite_mask):
                finite_params = primary_params[finite_mask]
                finite_costs_plot = costs[finite_mask]
                finite_steps_plot = steps[finite_mask]
                scatter2 = ax4.scatter(finite_params, finite_costs_plot, c=finite_steps_plot, 
                                     cmap='plasma', s=60, alpha=0.8, edgecolors='black', linewidth=0.5,
                                     label='Finite Cost')
                ax4.plot(finite_params, finite_costs_plot, 'k-', alpha=0.3, linewidth=1)
                
                # Add colorbar
                cbar2 = plt.colorbar(scatter2, ax=ax4)
                cbar2.set_label('Optimization Step')
                
                # Plot infinite costs as special markers
                if np.any(infinite_mask):
                    infinite_params = primary_params[infinite_mask]
                    max_finite_cost = np.max(finite_costs_plot)
                    infinite_plot_height = max_finite_cost * 1.3 if max_finite_cost > 0 else 1000
                    
                    ax4.scatter(infinite_params, [infinite_plot_height] * len(infinite_params),
                               marker='X', s=120, color='red', edgecolor='darkred', 
                               linewidth=2, label='Infinite Cost', zorder=4)
                    
                    # Add infinity symbols
                    for param in infinite_params:
                        ax4.annotate('∞', (param, infinite_plot_height), xytext=(0, 10), 
                                   textcoords='offset points', ha='center', va='bottom',
                                   fontsize=12, color='darkred', weight='bold')
            else:
                # Fallback if no finite costs
                ax4.scatter(primary_params, [1000] * len(primary_params), marker='X', s=120, color='red', 
                           edgecolor='darkred', linewidth=2, label='Infinite Cost')
                for param in primary_params:
                    ax4.annotate('∞', (param, 1000), xytext=(0, 10), 
                               textcoords='offset points', ha='center', va='bottom',
                               fontsize=12, color='darkred', weight='bold')
            
            ax4.axhline(y=0, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Match')
            ax4.set_xlabel(f'{param_name}')
            ax4.set_ylabel('Cost Function Value')
            ax4.set_title(f'Cost Landscape: {param_name} vs Cost')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save to output directory if provided, otherwise current directory
            # Include spinup_id in filename if provided
            if spinup_id:
                filename = f'optimization_progress_{str(spinup_id)}.png'
            else:
                filename = 'optimization_progress.png'
                
            if output_dir:
                plot_path = Path(output_dir) / filename
            else:
                plot_path = filename
            
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Optimization progress plot saved to: {plot_path}")
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating optimization plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_simulation_with_monitoring(self, output_dir, run_id, no_progress):
        """
        Run simulation with inline monitoring for target matching.
        
        This method creates a flowline2d model with extended simulation time
        and monitors it for steady-state achievement and cost evaluation.
        
        Parameters
        ----------
        output_dir : Path
            Directory for simulation output
        run_id : str
            Unique identifier for this simulation
        no_progress : bool
            Disable progress bars
            
        Returns
        -------
        float
            Final cost value when steady state is achieved
        """
        # Import here to avoid circular imports
        from flowline.flowline2d import flowline2d
        
        # Create extended config for optimization
        extended_config = deepcopy(self.config)
        if not self.target_matching:
            # Should not happen if called from optimization, but as a safeguard
            return float('inf')
        max_time = self.target_matching['max_simulation_time']
        extended_config.tf = max_time
        extended_config.deltout = 1.0  # Output every year for monitoring
        
        optimization_geometry = deepcopy(self.geometry)
        # Verify the copy:
        assert id(optimization_geometry) != id(self.geometry)
        assert id(optimization_geometry.zb_gr) != id(self.geometry.zb_gr)

        # Create model with optimization hooks
        # Pass self.forcing directly to ensure the updated T0 is used
        model = flowline2d(extended_config, optimization_geometry, self.forcing)
        
        # Run simulation with custom monitoring loop
        print("Starting custom monitoring loop for optimization...")
        cost = self._run_custom_monitoring_loop(
            model,
            no_progress,
            cost_function=self.cost_function,
            steady_state_detector=self.steady_state_detector,
            targets=self.target_matching['targets'],
            check_interval=self.target_matching.get('check_interval', 50)
        )
        print(f"Custom monitoring loop completed with cost: {cost}")
        return cost
    
    def _run_custom_monitoring_loop(self, model, no_progress, cost_function, steady_state_detector, targets, check_interval):
        """
        Run simulation with custom monitoring that can terminate early.
        
        This method steps through the simulation manually, checking for steady state
        at each output step and terminating early when steady state is achieved.
        
        Parameters
        ----------
        model : flowline2d
            The flowline2d model instance
        no_progress : bool
            Disable progress bars
        cost_function: CostFunction
            The cost function to evaluate.
        steady_state_detector: SteadyStateDetector
            The detector to check for steady state.
        targets: dict
            The target values for the cost function.
        check_interval: int
            The interval in years to check for steady state.
            
        Returns
        -------
        float
            Final cost value when steady state is achieved
        """
        from flowline.flowline2d import space_loop
        
        # Initialize simulation state
        # Initialize simulation state
        yr = model.config.ts - 1
        idx_out = 0
        t_out = 0.0
        t = 0.0
        b = np.zeros(model.nxs)
        climate_vars = {}
        h = model.h0.copy()
        
        # Initialize monitoring state
        time_history = []
        volume_history = []
        length_history = []
        optimization_cost = 1e12  # Use a large float instead of inf
        last_steady_state_check = 0.0

        # Setup progress bar
        if no_progress:
            range_iter = range(0, model.nts)
        else:
            range_iter = tqdm(
                range(0, model.nts),
                unit_scale=model.config.delt,
                bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
                unit="yrs",
                dynamic_ncols=True
            )
        
        for i in range_iter:
            t = model.config.delt * i
            
            # Update climate on integer year change
            current_model_year = model.config.ts + int(np.floor(t))
            if current_model_year > yr:
                yr = current_model_year
                year_idx = yr - model.config.ts
                
                # Calculate effective height for mass balance
                if model.config.hmb:
                    h_eff = model.zb + h
                else:
                    h_eff = model.zb
                
                # Get mass balance from forcing
                b, climate_vars = model.forcing.get_mass_balance(
                    model.x, h_eff, year_idx
                )
            
            # Solve shallow ice approximation
            h, edge_idx, F = space_loop(
                h, b, model.x, model.config.rho, model.config.g, model.nxs,
                model.config.delx, model.dzbdx, model.config.fd, model.config.fs,
                model.dwdx, model.w, model.config.delt, model.config.min_thick,
                model.config.n, model.config.k
            )
            
            # Check for numerical instability with detailed diagnostics
            if np.any(np.isnan(h)):
                # Capture detailed state for debugging
                nan_count = np.sum(np.isnan(h))
                h_min, h_max = np.nanmin(h), np.nanmax(h)
                h_mean = np.nanmean(h)
                
                # Check other variables for issues
                b_stats = f"b: min={np.min(b):.2f}, max={np.max(b):.2f}, mean={np.mean(b):.2f}"
                
                # Climate variables
                climate_info = []
                for var, val in climate_vars.items():
                    if val is None:
                        climate_info.append(f"{var}=None")
                    elif np.isscalar(val):
                        climate_info.append(f"{var}={val:.3f}")
                    else:
                        try:
                            climate_info.append(f"{var}: min={np.min(val):.3f}, max={np.max(val):.3f}")
                        except:
                            climate_info.append(f"{var}=invalid_array")
                
                # Model configuration
                config_info = f"delx={model.config.delx}, delt={model.config.delt:.6f}"
                
                error_msg = (
                    f"Numerical instability at t={t:.2f} years:\n"
                    f"  NaN values in h: {nan_count}/{len(h)} cells\n"
                    f"  h stats: min={h_min:.2f}, max={h_max:.2f}, mean={h_mean:.2f}\n"
                    f"  {b_stats}\n"
                    f"  Climate: {', '.join(climate_info)}\n"
                    f"  Config: {config_info}\n"
                    f"  edge_idx: {edge_idx}"
                )
                
                # Log to optimization logger if available
                if hasattr(self, 'opt_logger'):
                    self.opt_logger.error(error_msg)
                
                raise RuntimeError(error_msg)
            
            # Save output and check for steady state at specified interval
            if t >= t_out and idx_out < len(model.t):
                # Save output
                model._save_output(idx_out, t, h, b, edge_idx, F, climate_vars)
                
                # Update monitoring data (inline for efficiency)
                current_time = t + model.config.ts
                time_history.append(current_time)
                volume = np.sum(h[:edge_idx] * model.w[:edge_idx]) * model.config.delx if edge_idx > 0 else 0.0
                volume_history.append(volume)
                current_length = edge_idx * model.config.delx
                length_history.append(current_length)
                
                # Check for steady state at a fixed interval
                time_since_last_check = current_time - last_steady_state_check

                if time_since_last_check >= check_interval:
                    steady_state_achieved, cost = self._check_steady_state(
                        model, current_time, h, b, edge_idx,
                        cost_function, steady_state_detector, targets,
                        time_history, volume_history, length_history
                    )
                    if cost is not None:
                        optimization_cost = cost
                    if steady_state_achieved:
                        safe_print(f"*** EARLY TERMINATION *** Steady state achieved, terminating at t={t:.1f} years")
                        break
                    last_steady_state_check = current_time
                
                idx_out += 1
                t_out += model.config.deltout
        
        # Return final cost
        if optimization_cost >= 1e12:
            print("WARNING: Cost is very high! Simulation may not have reached steady state.")
            print(f"Final simulation time: {t:.1f} years")
            print(f"Number of monitoring history entries: {len(time_history)}")
            # Store final state so visualization shows actual (not zero) length
            final_volume = np.sum(h[:edge_idx] * model.w[:edge_idx]) * model.config.delx if edge_idx > 0 else 0.0
            self._last_optimization_state = {
                'final_length': edge_idx * model.config.delx,
                'final_edge_idx': edge_idx,
                'steady_state_time': f"{t:.1f}yr (no convergence)",
                'final_volume': final_volume
            }

        return optimization_cost
    
    def _check_steady_state(self, model, current_time, h, b, edge_idx, cost_function, steady_state_detector, targets, time_history, volume_history, length_history):
        """
        Check for steady state and calculate cost.
        
        Returns a tuple (steady_state_achieved, cost).
        """
        model_state_history = {
            'time_history': time_history,
            'volume_history': volume_history,
            'length_history': length_history
        }
        
        # Add more logging for steady state detection diagnostics
        if hasattr(self, 'opt_logger') and len(volume_history) >= 5:
            # Calculate recent dV/dt for diagnostics
            recent_volumes = np.array(volume_history[-5:])
            recent_times = np.array(time_history[-5:])
            if len(recent_times) > 1:
                dt = np.diff(recent_times)
                dV = np.diff(recent_volumes)
                valid_mask = dt > 0
                if np.any(valid_mask):
                    dV_dt = dV[valid_mask] / dt[valid_mask]
                    self.opt_logger.debug(f"Steady state check: dV/dt history = {dV_dt}")
            
            # Log edge position history  
            recent_lengths = np.array(length_history[-5:])
            self.opt_logger.debug(f"Edge position history = {recent_lengths}")
        
        glacier_melted = edge_idx <= 0
        
        if steady_state_detector(model_state_history) or glacier_melted:
            volume = np.sum(h[:edge_idx] * model.w[:edge_idx]) * model.config.delx if edge_idx > 0 else 0.0
            final_length = edge_idx * model.config.delx
            
            if hasattr(self, 'opt_logger'):
                if glacier_melted:
                    self.opt_logger.info(f"STEADY STATE: Glacier completely melted at t={current_time:.1f} years")
                else:
                    self.opt_logger.info(f"STEADY STATE: Detector triggered at t={current_time:.1f} years, length={final_length:.1f}m, volume={volume:.1f}")
            
            safe_print(f"*** STEADY STATE DETECTED at t={current_time:.1f} years ***")
            
            final_model_state = {
                'edge': edge_idx,
                'delx': self.config.delx,
                'b': b,
                'ela': model.ela[-1],
                'F': model.F[-1],
                'h': h,
                'volume': volume,
                'area': np.sum(model.w[:edge_idx]) * self.config.delx if edge_idx > 0 else 0.0
            }
            
            if hasattr(self, 'opt_logger'):
                cost_function._debug_logger = self.opt_logger
            
            cost = cost_function(final_model_state, targets)
            
            if hasattr(self, 'opt_logger'):
                target_length = targets.get('target_length', 'N/A')
                self.opt_logger.info(f"COST CALCULATION: Target length={target_length}m, actual length={final_length:.1f}m, cost={cost:.6f}")
            
            self._last_optimization_state = {
                'final_length': final_length,
                'final_edge_idx': edge_idx,
                'steady_state_time': f"{current_time:.1f}yr",
                'final_volume': volume
            }
            
            return True, cost
        
        return False, None
    
    def _run_steady_state_spinup(self, output_dir, spinup_id, no_progress):
        """
        Execute the spinup run to generate steady-state profile.
        
        Returns
        -------
        str
            Path to the generated steady-state profile
        """
        spinup_geometry = deepcopy(self.geometry)

        # Run the spinup simulation
        result = run_spinup_simulation(
            (spinup_id, self.config, spinup_geometry, self.forcing, output_dir, no_progress)
        )
        
        if str(result).startswith("ERROR"):
            raise RuntimeError(f"Spinup failed for {spinup_id}: {result}")
        
        print(f"Spinup completed for {spinup_id}: {result}")
        return result
    
    
    def __eq__(self, other):
        """
        Check equality for object sharing in FlowlineSweep.
        
        Two FlowlineSpinup objects are equal if they have the same configuration.
        This allows FlowlineSweep to identify when multiple run_ids can share
        the same spinup object.
        """
        if not isinstance(other, FlowlineSpinup):
            return False
        
        return (
            objects_equal(self.config, other.config) and
            objects_equal(self.geometry, other.geometry) and
            objects_equal(self.forcing, other.forcing) and
            self.target_matching == other.target_matching
        )
    
    def __hash__(self):
        """
        Make FlowlineSpinup hashable for use in dictionaries and sets.
        """
        return hash((
            object_hash(self.config),
            object_hash(self.geometry),
            object_hash(self.forcing),
            str(sorted(self.target_matching.items())) if self.target_matching else ""
        ))
    
    def _log_forcing_state(self, forcing, label="FORCING"):
        """
        Log the current state of a forcing object for debugging.
        
        Parameters
        ----------
        forcing : MassBalanceForcing
            The forcing object to log
        label : str
            Label for the log messages
        """
        if not hasattr(self, 'opt_logger'):
            return
            
        try:
            # Log key forcing parameters
            key_params = ['T0', 'P0', 'mu', 'gamma', 'ts', 'tf']
            param_values = []
            
            for param in key_params:
                if hasattr(forcing, param):
                    value = getattr(forcing, param)
                    param_values.append(f"{param}={value}")
                else:
                    param_values.append(f"{param}=N/A")
            
            self.opt_logger.debug(f"{label}: {', '.join(param_values)}")
            
            # Log object ID for reference tracking
            self.opt_logger.debug(f"{label}_ID: {id(forcing)}")
            
        except Exception as e:
            self.opt_logger.debug(f"Could not log forcing state: {e}")
    
