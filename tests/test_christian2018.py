"""
Test configuration to approximate Glacier 1 from Christian et al. (2018)
"Committed retreat: controls on glacier disequilibrium in a warming climate"

This configuration matches the smaller, steeper glacier from the paper:
- Bed slope: tan(φ) = 0.2 (11.3°)
- Length: ~6.55 km
- Response time: ~25 years
- Mean thickness: ~54 m
"""

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import your flowline module
import sys
sys.path.append('src')
from flowline.flowline2d import (
    flowline2d, FlowlineConfig, FlowlineGeometry, 
    TemperaturePrecipitationForcing, DirectMassBalanceForcing
)

class TestChristianGlacier1:
    """Test configuration matching Glacier 1 from Christian et al. (2018)"""
    
    @pytest.fixture
    def christian_glacier1_params(self):
        """Parameters for Glacier 1 from Christian et al. (2018)"""
        return {
            # Geometry parameters
            'max_elevation': 2500,  # m a.s.l.
            'bed_slope': 0.2,  # tan(φ) = 0.2 (11.3°)
            'domain_length': 8000,  # m (slightly longer than steady-state length)
            'x_gr_points': 41,  # Similar to your existing tests
            
            # Climate parameters (from Table 1)
            'T0': 20,  # °C melt-season temp at sea level
            'P0': 4,   # m/yr accumulation (ice equivalent)
            'gamma': 6.5e-3,  # °C/m lapse rate
            'mu': 0.5,  # m/yr/°C melt factor
            
            # Expected steady-state properties
            'expected_length': 6550,  # m (6.55 km from paper)
            'expected_thickness': 54,  # m mean thickness
            'expected_response_time': 25,  # years
            'expected_terminus_balance': -2.12,  # m/yr (ice equiv.)
        }
    
    def create_christian_glacier1_geometry(self, params):
        """Create geometry matching Christian et al. Glacier 1"""
        # Create domain
        x_gr = np.linspace(0, params['domain_length'], params['x_gr_points'])
        
        # Bed elevation: linear slope from max_elevation to sea level
        # Slope = elevation_drop / horizontal_distance
        # For bed slope tan(φ) = 0.2, and max elevation 2500m:
        # horizontal_distance = elevation / tan(φ) = 2500 / 0.2 = 12500 m
        bed_length = params['max_elevation'] / params['bed_slope']
        
        # Bed elevation profile
        zb_gr = np.maximum(0, params['max_elevation'] - params['bed_slope'] * x_gr)
        
        # Constant width (paper doesn't specify, using reasonable value)
        width = 1000  # m (1 km width)
        w_geom = np.full_like(x_gr, width)
        
        return x_gr, zb_gr, w_geom
    
    def test_christian_glacier1_steady_state(self, christian_glacier1_params):
        """Test steady-state configuration matching Christian et al. Glacier 1"""
        params = christian_glacier1_params
        
        # Configuration for steady-state run
        config = FlowlineConfig(
            delx=25,  # 25m grid spacing (from paper)
            delt=0.0125/32,  # Small timestep for stability
            ts=0,
            tf=500,  # Run for 500 years to reach steady state
            deltout=1,  # Output every 10 years
        )
        
        # Create geometry
        x_gr, zb_gr, w_geom = self.create_christian_glacier1_geometry(params)
        
        # Initial thickness - triangular profile
        initial_length = 5000  # Start with 5 km glacier
        h_init = np.maximum(0, 100 * (1 - x_gr / initial_length))
        
        # Create geometry object
        geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, x_gr, h_init)
        
        # Temperature-precipitation forcing
        forcing = TemperaturePrecipitationForcing(
            T0=params['T0'],
            P0=params['P0'],
            gamma=params['gamma'],
            mu=params['mu'],
            ts=config.ts,
            tf=config.tf
        )
        
        # Run model
        model = flowline2d(config=config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create QC figure
        self._create_christian_comparison_figure(result, params, 
                                                'christian_glacier1_steady_state.png')
        
        # Check steady-state properties
        final_length = result.edge[-1]
        final_edge_idx = result.edge_idx[-1]
        
        if final_edge_idx > 0:
            mean_thickness = np.mean(result.h[-1, :final_edge_idx])
            
            # Calculate terminus mass balance
            terminus_elevation = result.zb[final_edge_idx-1]
            terminus_temp = params['T0'] - params['gamma'] * terminus_elevation
            terminus_balance = params['P0'] - params['mu'] * terminus_temp
            
            # Calculate response time
            response_time = mean_thickness / abs(terminus_balance)
            
            print(f"Final glacier length: {final_length/1000:.2f} km (expected: {params['expected_length']/1000:.2f} km)")
            print(f"Mean thickness: {mean_thickness:.1f} m (expected: {params['expected_thickness']:.1f} m)")
            print(f"Terminus balance: {terminus_balance:.2f} m/yr (expected: {params['expected_terminus_balance']:.2f} m/yr)")
            print(f"Response time: {response_time:.1f} years (expected: {params['expected_response_time']:.1f} years)")
            
            # Assertions with reasonable tolerances
            assert abs(final_length - params['expected_length']) / params['expected_length'] < 0.2, \
                f"Length error: {abs(final_length - params['expected_length'])/1000:.2f} km"
            
            assert abs(mean_thickness - params['expected_thickness']) / params['expected_thickness'] < 0.3, \
                f"Thickness error: {abs(mean_thickness - params['expected_thickness']):.1f} m"
            
            assert abs(response_time - params['expected_response_time']) / params['expected_response_time'] < 0.4, \
                f"Response time error: {abs(response_time - params['expected_response_time']):.1f} years"
        
        return result
    
    def test_christian_glacier1_warming_response(self, christian_glacier1_params):
        """Test glacier response to gradual warming as in Christian et al."""
        params = christian_glacier1_params
        
        # First get steady-state
        steady_result = self.test_christian_glacier1_steady_state(params)
        
        # Configuration for warming experiment
        warming_config = FlowlineConfig(
            delx=25,
            delt=0.0125/32,
            ts=0,
            tf=200,  # 200 year warming period (as in paper)
            deltout=1,  # Output every 2 years
        )
        
        # Create linear warming trend: +2°C over 200 years
        nyears = int(warming_config.tf - warming_config.ts)
        temperature_trend = np.linspace(0, 2, nyears)  # 0 to +2°C
        
        # Base mass balance from steady state
        ss_b_profile = steady_result.b_profile[-1, :]
        
        # Mass balance perturbation due to warming
        # Δb = -μ * ΔT (negative because warming increases melt)
        bp_warming = -params['mu'] * temperature_trend
        
        # Direct mass balance forcing
        forcing = DirectMassBalanceForcing(
            b0=ss_b_profile,
            bp=bp_warming
        )
        
        # Geometry from steady state
        geometry = FlowlineGeometry(
            steady_result.x_gr, steady_result.zb_gr, steady_result.w_geom, 
            profile=steady_result
        )
        
        # Run warming experiment
        model = flowline2d(config=warming_config, geometry=geometry, forcing=forcing)
        result = model.run()
        
        # Create comparison figure
        self._create_warming_response_figure(result, params, 
                                           'christian_glacier1_warming_response.png')
        
        # Calculate disequilibrium metrics
        initial_length = result.edge[0]
        final_length = result.edge[-1]
        
        # Equilibrium length for +2°C warming
        # From paper's linear model: ΔL = β * τ * Δb
        # where β = L/H and τ = H/|bt|
        mean_thickness = params['expected_thickness']
        terminus_balance = abs(params['expected_terminus_balance'])
        beta = params['expected_length'] / mean_thickness
        tau = mean_thickness / terminus_balance
        
        delta_b = -params['mu'] * 2  # -1 m/yr for +2°C
        equilibrium_retreat = beta * tau * abs(delta_b)
        equilibrium_length = initial_length - equilibrium_retreat
        
        # Fractional equilibration
        actual_retreat = initial_length - final_length
        fractional_equilibration = actual_retreat / equilibrium_retreat
        
        print(f"Initial length: {initial_length/1000:.2f} km")
        print(f"Final length: {final_length/1000:.2f} km")
        print(f"Actual retreat: {actual_retreat/1000:.2f} km")
        print(f"Equilibrium retreat: {equilibrium_retreat/1000:.2f} km")
        print(f"Fractional equilibration: {fractional_equilibration:.2f}")
        
        # From Figure 3e in paper, Glacier 1 should be ~75% equilibrated after 200 years
        expected_equilibration = 0.75
        assert abs(fractional_equilibration - expected_equilibration) < 0.2, \
            f"Equilibration error: {fractional_equilibration:.2f} vs expected {expected_equilibration:.2f}"
        
        return result
    
    def _create_christian_comparison_figure(self, result, params, filename):
        """Create QC figure comparing with Christian et al. results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Christian et al. (2018) Glacier 1 - Model Comparison', fontsize=14)
        
        # Plot 1: Final ice thickness profile
        ax = axes[0, 0]
        edge_idx = result.edge_idx[-1]
        if edge_idx > 0:
            ax.fill_between(result.x[:edge_idx]/1000, result.zb[:edge_idx], 
                           result.zb[:edge_idx] + result.h[-1, :edge_idx], 
                           alpha=0.7, color='lightblue', label='Ice')
        ax.plot(result.x/1000, result.zb, 'k-', linewidth=2, label='Bed')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Final Ice Thickness Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add expected length line
        ax.axvline(x=params['expected_length']/1000, color='red', linestyle='--', 
                  label=f"Expected length ({params['expected_length']/1000:.1f} km)")
        ax.legend()
        
        # Plot 2: Length evolution
        ax = axes[0, 1]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=2)
        ax.axhline(y=params['expected_length']/1000, color='red', linestyle='--', 
                  label=f"Expected length ({params['expected_length']/1000:.1f} km)")
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Evolution to Steady State')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Parameter comparison
        ax = axes[1, 0]
        final_length = result.edge[-1]
        final_edge_idx = result.edge_idx[-1]
        
        if final_edge_idx > 0:
            mean_thickness = np.mean(result.h[-1, :final_edge_idx])
            
            # Calculate actual values
            terminus_elevation = result.zb[final_edge_idx-1]
            terminus_temp = params['T0'] - params['gamma'] * terminus_elevation
            terminus_balance = params['P0'] - params['mu'] * terminus_temp
            response_time = mean_thickness / abs(terminus_balance)
            
            # Comparison data
            parameters = ['Length (km)', 'Thickness (m)', 'Response Time (yr)', 'Terminus Balance (m/yr)']
            expected = [params['expected_length']/1000, params['expected_thickness'], 
                       params['expected_response_time'], params['expected_terminus_balance']]
            actual = [final_length/1000, mean_thickness, response_time, terminus_balance]
            
            x_pos = np.arange(len(parameters))
            width = 0.35
            
            ax.bar(x_pos - width/2, expected, width, label='Expected (Paper)', alpha=0.7)
            ax.bar(x_pos + width/2, actual, width, label='Model Result', alpha=0.7)
            
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Values')
            ax.set_title('Parameter Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(parameters, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Mass balance profile
        ax = axes[1, 1]
        if hasattr(result, 'b_profile') and result.b_profile is not None:
            if edge_idx > 0:
                ax.plot(result.x[:edge_idx]/1000, result.b_profile[-1, :edge_idx], 'r-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Mass Balance (m/yr)')
                ax.set_title('Mass Balance Profile')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path("test_qc_figures") / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_warming_response_figure(self, result, params, filename):
        """Create figure showing warming response"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Christian et al. (2018) Glacier 1 - Warming Response', fontsize=14)
        
        # Plot 1: Length evolution
        ax = axes[0, 0]
        ax.plot(result.t, result.edge/1000, 'b-', linewidth=2, label='Transient')
        
        # Calculate equilibrium response
        initial_length = result.edge[0]
        mean_thickness = params['expected_thickness']
        terminus_balance = abs(params['expected_terminus_balance'])
        beta = params['expected_length'] / mean_thickness
        tau = mean_thickness / terminus_balance
        
        # Equilibrium length for each time
        warming_rate = 2.0 / 200  # °C/year
        equilibrium_lengths = []
        for t in result.t:
            total_warming = warming_rate * t
            delta_b = -params['mu'] * total_warming
            equilibrium_retreat = beta * tau * abs(delta_b)
            eq_length = initial_length - equilibrium_retreat
            equilibrium_lengths.append(eq_length)
        
        ax.plot(result.t, np.array(equilibrium_lengths)/1000, 'r--', linewidth=2, label='Equilibrium')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Glacier Length (km)')
        ax.set_title('Length Response to 2°C Warming over 200 years')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Disequilibrium
        ax = axes[0, 1]
        disequilibrium = (result.edge - np.array(equilibrium_lengths)) / 1000
        ax.plot(result.t, disequilibrium, 'g-', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Disequilibrium (km)')
        ax.set_title('Glacier Disequilibrium')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fractional equilibration
        ax = axes[1, 0]
        initial_length = result.edge[0]
        actual_retreat = initial_length - result.edge
        equilibrium_retreat = initial_length - np.array(equilibrium_lengths)
        
        # Avoid division by zero
        fractional_eq = np.where(equilibrium_retreat != 0, actual_retreat / equilibrium_retreat, 0)
        ax.plot(result.t, fractional_eq, 'purple', linewidth=2)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Fractional Equilibration')
        ax.set_title('Fractional Equilibration')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add 140-year mark (current state)
        ax.axvline(x=140, color='orange', linestyle=':', label='~Current (140 yr)')
        ax.legend()
        
        # Plot 4: Final profiles comparison
        ax = axes[1, 1]
        edge_idx = result.edge_idx[-1]
        if edge_idx > 0:
            ax.fill_between(result.x[:edge_idx]/1000, result.zb[:edge_idx], 
                           result.zb[:edge_idx] + result.h[-1, :edge_idx], 
                           alpha=0.7, color='lightblue', label='Final ice')
        ax.plot(result.x/1000, result.zb, 'k-', linewidth=2, label='Bed')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Final Ice Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path("tests/qc_figures") / filename, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Run the Christian et al. Glacier 1 tests
    test_instance = TestChristianGlacier1()
    
    # Create parameters
    params = {
        'max_elevation': 2500,
        'bed_slope': 0.2,
        'domain_length': 8000,
        'x_gr_points': 41,
        'T0': 20,
        'P0': 4,
        'gamma': 6.5e-3,
        'mu': 0.5,
        'expected_length': 6550,
        'expected_thickness': 54,
        'expected_response_time': 25,
        'expected_terminus_balance': -2.12,
    }
    
    print("Testing Christian et al. (2018) Glacier 1 configuration...")
    print("=" * 60)
    
    # Test steady state
    print("1. Testing steady-state configuration...")
    steady_result = test_instance.test_christian_glacier1_steady_state(params)
    print("✓ Steady-state test completed")
    print()
    
    # Test warming response
    print("2. Testing warming response...")
    warming_result = test_instance.test_christian_glacier1_warming_response(params)
    print("✓ Warming response test completed")
    print()
    
    print("All tests completed! Check tests/qc_figures/ for output plots.")
