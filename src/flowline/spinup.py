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
from scipy.optimize import minimize_scalar
from flowline.entrypoints import run_spinup_simulation
from flowline.utils import objects_equal, object_hash
from tqdm import tqdm

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


class LengthOnlyCost(CostFunction):
    """Cost function that only considers glacier length."""
    
    def __call__(self, model_state: Dict[str, Any], targets: Dict[str, Any]) -> float:
        edge_idx = model_state['edge']
        current_length = edge_idx * model_state['delx']
        target_length = targets['target_length']
        error = current_length - target_length
        
        if math.isinf(error):
            error = 1e8  # arbitrarily high number
        
        return abs(error)

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
    
    def __init__(self, threshold=1e2, window_size=10, min_time=100):
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

def calculate_response_time(h, b, delx, edge_idx, ela_idx, **kwargs):
    """
    Calculate glacier response time tau using mass balance and thickness.
    
    Calculates the mean thickness between ELA and terminus, and uses the 
    mass balance value at the terminus.
    
    Parameters
    ----------
    h : array_like
        Ice thickness array (m)
    b : array_like
        Mass balance array (m/year)
    delx : float
        Grid spacing (m)
    edge_idx : int
        Index of glacier terminus
    ela_idx : int
        Index of ELA
    **kwargs : dict
        Additional parameters for response time calculation
        
    Returns
    -------
    float
        Response time in years
        
    Raises
    ------
    ValueError
        If glacier is completely melted (edge_idx <= 0) or calculation fails
    """
    # Handle case where glacier is completely melted (edge_idx = 0)
    if edge_idx <= 0:
        raise ValueError("Cannot calculate response time for completely melted glacier")
    
    # Handle case where ELA is beyond terminus (ela_idx >= edge_idx)
    if ela_idx >= edge_idx:
        raise ValueError("ELA is beyond glacier terminus")
    
    # Calculate mean thickness between ELA and terminus
    thickness_slice = h[ela_idx:edge_idx]
    if len(thickness_slice) == 0:
        raise ValueError("Empty thickness slice for response time calculation")
    
    mean_thickness = np.mean(thickness_slice)
    
    # Get mass balance at terminus (edge_idx-1 to avoid index error)
    terminus_idx = max(0, edge_idx - 1)
    terminus_mass_balance = b[terminus_idx]
    
    # Calculate response time: tau = -H / b_terminus
    if terminus_mass_balance == 0:
        raise ValueError("Terminus mass balance is zero")
    
    tau = -mean_thickness / terminus_mass_balance
    
    # Check for invalid results
    if not np.isfinite(tau):
        raise ValueError("Response time calculation resulted in non-finite value")
    
    if tau <= 0:
        raise ValueError("Response time must be positive")
    
    return tau








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
        spec = self.target_matching['cost_function']
        return spec() if isinstance(spec, type) else spec
    
    @property 
    def steady_state_detector(self):
        """Get steady-state detector, instantiating if needed.
        
        Accesses target_matching['steady_state_detector'] and instantiates it if it's a class.
        Will fail with KeyError if target_matching is missing or doesn't contain 'steady_state_detector'.
        This is intentional - no silent defaults.
        """
        spec = self.target_matching['steady_state_detector']
        return spec() if isinstance(spec, type) else spec
    
    
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
        
        optimized_parameters = {}
        if self.target_matching:
            # Use optimization-based target matching
            optimized_param = self._optimize_target_matching(output_dir, spinup_id, no_progress)
            safe_print(f"Target matching optimization completed. Optimal {self.target_matching['adjustment_parameter']} = {optimized_param:.3f}")
            # Store the optimized parameter
            optimized_parameters[self.target_matching['adjustment_parameter']] = optimized_param
        
        # Run final spinup with optimized parameters
        profile_path = self._run_steady_state_spinup(output_dir, spinup_id, no_progress)
        
        return profile_path, optimized_parameters
    
    def _optimize_target_matching(self, output_dir, spinup_id, no_progress):
        """
        Optimize parameter using minimize_scalar to achieve target matching.
        
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
        float
            Optimal parameter value
        """
        adjustment_parameter = self.target_matching['adjustment_parameter']
        targets = self.target_matching['targets']
        tolerance = self.target_matching["tolerance"]
        max_iterations = self.target_matching["max_iterations"]
        
        # Initialize optimization history tracking
        self._optimization_history = {'params': [], 'costs': [], 'lengths': [], 'states': []}
        
        def objective_function(param_value):
            """
            Objective function for optimization.
            
            Runs simulation with given parameter value and returns cost.
            """
            # Set the parameter value
            setattr(self.forcing, adjustment_parameter, param_value)
            print(f"Optimization trying: {adjustment_parameter} = {param_value:.3f}")
            
            # Run simulation with inline optimization monitoring
            cost = self._run_simulation_with_monitoring(
                output_dir, f"{spinup_id}_opt_{param_value:.3f}", no_progress
            )
            
            print(f"Optimization result: {adjustment_parameter} = {param_value:.3f}, cost = {cost:.1f}")
            
            # Add diagnostic info about the final state and store optimization history
            if hasattr(self, '_last_optimization_state'):
                state = self._last_optimization_state
                final_length = state["final_length"]
                final_edge_idx = state["final_edge_idx"]
                steady_state_time = state["steady_state_time"]
                safe_print(f"  -> Final length: {final_length:.1f}m (edge_idx={final_edge_idx}), steady state at: {steady_state_time}")
                
                # Store in optimization history
                self._optimization_history['params'].append(param_value)
                self._optimization_history['costs'].append(cost)
                self._optimization_history['lengths'].append(final_length)
                self._optimization_history['states'].append(state)
            else:
                safe_print("  -> No optimization state available")
                self._optimization_history['params'].append(param_value)
                self._optimization_history['costs'].append(cost)
                self._optimization_history['lengths'].append(0)
                self._optimization_history['states'].append({})
            
            return cost
        
        # Run optimization
        bounds = self.target_matching['parameter_bounds']
        result = minimize_scalar(
            objective_function,
            bounds=bounds,
            tol=tolerance,
            method='bounded',
            options={'maxiter': max_iterations, 'disp':True}
        )
        
        if not result.success:
            safe_print(f"Warning: Optimization did not converge. Using best result: {result.x:.3f}")
        
        # Set the optimal parameter value
        optimal_param = result.x
        setattr(self.forcing, adjustment_parameter, optimal_param)
        logger.debug(f"OPTIMIZATION: Set {adjustment_parameter}={optimal_param:.3f} on forcing object, now T0={self.forcing.T0:.3f}")
        
        # Create a simple visualization of the optimization progress
        self._create_optimization_plot(output_dir, spinup_id)
        
        return optimal_param
    
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
            mosaic = [
                ['step_temp', 'temp_length'],
                ['step_cost', 'temp_cost']
            ]
            axes = fig.subplot_mosaic(mosaic)
            
            params = np.array(self._optimization_history['params'])
            costs = np.array(self._optimization_history['costs'])
            lengths = np.array(self._optimization_history['lengths'])
            steps = np.arange(len(params))
            
            # Filter out infinite costs for some plots
            finite_mask = np.isfinite(costs)
            param_name = self.target_matching["adjustment_parameter"] if self.target_matching else "Parameter"
            target_length = self.target_matching['targets']['target_length'] if self.target_matching else 8000
            
            # Plot 1: Optimization Step vs Temperature (shows path optimizer takes)
            ax1 = axes['step_temp']
            ax1.plot(steps, params, 'b-o', linewidth=2, markersize=6, label=f'{param_name} Path')
            ax1.set_xlabel('Optimization Step')
            ax1.set_ylabel(f'{param_name} (°C)')
            ax1.set_title('Optimization Path: Step vs Temperature')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Temperature vs Length (relationship)
            ax2 = axes['temp_length']
            scatter = ax2.scatter(params, lengths, c=steps, cmap='viridis', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax2.plot(params, lengths, 'k-', alpha=0.3, linewidth=1)
            ax2.axhline(y=target_length, color='r', linestyle='--', linewidth=2, label=f'Target ({target_length}m)')
            ax2.set_xlabel(f'{param_name} (°C)')
            ax2.set_ylabel('Glacier Length (m)')
            ax2.set_title('Temperature vs Length Relationship')
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
            
            # Plot 4: Temperature vs Cost (cost landscape)
            ax4 = axes['temp_cost']
            
            # Plot finite costs with color-coded steps
            if np.any(finite_mask):
                finite_params = params[finite_mask]
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
                    infinite_params = params[infinite_mask]
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
                ax4.scatter(params, [1000] * len(params), marker='X', s=120, color='red', 
                           edgecolor='darkred', linewidth=2, label='Infinite Cost')
                for param in params:
                    ax4.annotate('∞', (param, 1000), xytext=(0, 10), 
                               textcoords='offset points', ha='center', va='bottom',
                               fontsize=12, color='darkred', weight='bold')
            
            ax4.axhline(y=0, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Match')
            ax4.set_xlabel(f'{param_name} (°C)')
            ax4.set_ylabel('Cost Function Value')
            ax4.set_title('Cost Landscape: Temperature vs Cost')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save to output directory if provided, otherwise current directory
            # Include spinup_id in filename if provided
            if spinup_id:
                filename = f'optimization_progress_{spinup_id}.png'
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
        max_time = self.target_matching['max_simulation_time']
        extended_config.tf = max_time
        extended_config.deltout = 1.0  # Output every year for monitoring
        
        # Create geometry without existing profile
        optimization_geometry = deepcopy(self.geometry)
        if hasattr(optimization_geometry, 'profile'):
            optimization_geometry.profile = None
        
        # Create extended forcing
        extended_forcing = deepcopy(self.forcing)
        if hasattr(extended_forcing, 'tf'):
            extended_forcing.tf = max_time
        
        # Create model with optimization hooks
        model = flowline2d(extended_config, optimization_geometry, extended_forcing)
        
        # Add optimization components to model
        model.target_matching = self.target_matching
        model.cost_function = self.cost_function
        model.steady_state_detector = self.steady_state_detector
        model.optimization_targets = self.target_matching['targets']
        
        # Initialize monitoring state
        model.time_history = []
        model.volume_history = []
        model.length_history = []
        model.optimization_cost = float('inf')
        model.steady_state_achieved = False
        
        # Run simulation with custom monitoring loop
        print("Starting custom monitoring loop for optimization...")
        cost = self._run_custom_monitoring_loop(model, no_progress)
        print(f"Custom monitoring loop completed with cost: {cost}")
        return cost
    
    def _run_custom_monitoring_loop(self, model, no_progress):
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
            
        Returns
        -------
        float
            Final cost value when steady state is achieved
        """
        from flowline.flowline2d import space_loop
        
        # Initialize simulation state
        yr = model.config.ts - 1
        idx_out = 0
        t_out = 0.0
        b = np.zeros(model.nxs)
        climate_vars = {}
        h = model.h0.copy()
        
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
            
            # Check for numerical instability
            if np.any(np.isnan(h)):
                raise RuntimeError(f"Numerical instability at t={t:.2f} years")
            
            # Save output and check for steady state at specified interval
            if t >= t_out and idx_out < len(model.t):
                # Save output
                model._save_output(idx_out, t, h, b, edge_idx, F, climate_vars)
                
                # Check for steady state
                if self._check_steady_state_and_cost(model, t, h, b, edge_idx):
                    safe_print(f"*** EARLY TERMINATION *** Steady state achieved, terminating at t={t:.1f} years")
                    break
                
                idx_out += 1
                t_out += model.config.deltout
        
        # Return final cost
        if model.optimization_cost == float('inf'):
            print("WARNING: Cost is still infinite! Simulation may not have reached steady state.")
            print(f"Final simulation time: {t:.1f} years")
            print(f"Steady state achieved: {getattr(model, 'steady_state_achieved', False)}")
            print(f"Number of monitoring history entries: {len(model.time_history)}")
        
        return model.optimization_cost
    
    def _check_steady_state_and_cost(self, model, t, h, b, edge_idx):
        """
        Check for steady state and calculate cost.
        
        Returns True if steady state is achieved, False otherwise.
        """
        # Update monitoring history
        current_time = t + model.config.ts
        model.time_history.append(current_time)
        
        # Calculate volume
        volume = np.sum(h[:edge_idx] * model.w[:edge_idx]) * model.config.delx if edge_idx > 0 else 0.0
        model.volume_history.append(volume)
        
        # Store current length
        current_length = edge_idx * model.config.delx
        model.length_history.append(current_length)
        
        # Check for steady state
        model_state = {
            'time_history': model.time_history,
            'volume_history': model.volume_history,
            'length_history': model.length_history
        }
        
        # Special case: completely melted glacier is at steady state
        glacier_melted = edge_idx <= 0
        
        if self.steady_state_detector(model_state):
            if model.steady_state_achieved:
                
                # Calculate final cost
                final_model_state = {
                    'edge': edge_idx,
                    'delx': model.config.delx,
                    'h': h,
                    'volume': volume,
                    'area': np.sum(model.w[:edge_idx]) * model.config.delx if edge_idx > 0 else 0.0
                }
                
                model.optimization_cost = self.cost_function(final_model_state, self.target_matching['targets'])
                
                # Store diagnostic information for optimization output
                final_length = edge_idx * model.config.delx
                self._last_optimization_state = {
                    'final_length': final_length,
                    'final_edge_idx': edge_idx,
                    'steady_state_time': f"{current_time:.1f}yr",
                    'final_volume': volume
                }
                
                # safe_print(f"Steady state achieved at t={current_time:.1f} years")
                # safe_print(f"Final glacier length: {final_length:.1f}m")
                # safe_print(f"Final cost: {model.optimization_cost:.1f}")
                
                return True  # Steady state achieved
        
        return False  # Continue simulation
    
    def _run_steady_state_spinup(self, output_dir, spinup_id, no_progress):
        """
        Execute the spinup run to generate steady-state profile.
        
        Returns
        -------
        str
            Path to the generated steady-state profile
        """
        # Ensure geometry starts from h_init (no existing profile)
        spinup_geometry = deepcopy(self.geometry)
        if hasattr(spinup_geometry, 'profile'):
            spinup_geometry.profile = None
        
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
    
