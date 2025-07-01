import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class GlacierELAAnalysis:
    """
    A class to analyze glacier Equilibrium Line Altitude (ELA) 
    across different parameter ranges using xarray.
    """
    
    def __init__(self):
        self.dataset = None
        self.ela_dataset = None
        
    def calc_ela(self, P0, T0, gamma, mu, h=None):
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
    
    def calc_mass_balance(self, h, P0, T0, gamma, mu):
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
    
    def create_parameter_sweep(self, 
                             elev_range=(0, 4000, 50),
                             mu_range=(0.1, 1.5, 0.05), 
                             gamma_range=(4, 10, 0.5),
                             T0_range=(5, 20, 1),
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
                        mass_balance_data[i,j,k,l] = self.calc_mass_balance(
                            h, P0, T0, gamma, mu)
        
        # Calculate ELA for all parameter combinations (excluding elevation)
        for j, mu in enumerate(mu_vals):
            for k, gamma in enumerate(gamma_vals):
                for l, T0 in enumerate(T0_vals):
                    ela_data[j,k,l] = self.calc_ela(P0, T0, gamma, mu)
        
        # Create xarray datasets
        self.dataset = xr.Dataset({
            'mass_balance': (['elevation', 'mu', 'gamma', 'T0'], mass_balance_data),
            'P0': P0
        }, coords=coords)
        
        self.ela_dataset = xr.Dataset({
            'ELA': (['mu', 'gamma', 'T0'], ela_data),
            'P0': P0
        }, coords={k: v for k, v in coords.items() if k != 'elevation'})
        
        return self.dataset, self.ela_dataset
    
    def plot_mass_balance_profiles(self, mu_subset=None, gamma_subset=None, 
                                 T0_subset=None, figsize=(15, 10)):
        """Plot mass balance vs elevation for selected parameter combinations"""
        
        if self.dataset is None:
            raise ValueError("Must run create_parameter_sweep first")
            
        # Select subsets if provided
        data = self.dataset
        if mu_subset is not None:
            data = data.sel(mu=mu_subset, method='nearest')
        if gamma_subset is not None:
            data = data.sel(gamma=gamma_subset, method='nearest')
        if T0_subset is not None:
            data = data.sel(T0=T0_subset, method='nearest')
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Mass balance vs elevation for different mu values
        ax = axes[0]
        if 'mu' in data.dims and len(data.mu) > 1:
            mu_values = data.mu.values[::3]  # Every 3rd value to avoid crowding
            colors = cm.viridis(np.linspace(0, 1, len(mu_values)))
            for i, mu in enumerate(mu_values):
                mb_profile = data.sel(mu=mu, method='nearest').mass_balance
                if mb_profile.ndim > 1:
                    mb_profile = mb_profile.mean(dim=[d for d in mb_profile.dims if d != 'elevation'])
                ax.plot(data.elevation, mb_profile, label=f'μ={mu:.2f}', color=colors[i])
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero balance line')
            ax.set_title('Mass Balance vs Elevation (varying μ)')
            ax.legend()
        
        # Plot 2: Mass balance vs elevation for different gamma values  
        ax = axes[1]
        if 'gamma' in data.dims and len(data.gamma) > 1:
            gamma_values = data.gamma.values[::2]  # Every 2nd value
            colors = cm.plasma(np.linspace(0, 1, len(gamma_values)))
            for i, gamma in enumerate(gamma_values):
                mb_profile = data.sel(gamma=gamma, method='nearest').mass_balance
                if mb_profile.ndim > 1:
                    mb_profile = mb_profile.mean(dim=[d for d in mb_profile.dims if d != 'elevation'])
                ax.plot(data.elevation, mb_profile, label=f'γ={gamma:.1f}°C/km', color=colors[i])
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title('Mass Balance vs Elevation (varying γ)')
            ax.legend()
        
        # Plot 3: ELA across parameter space (if available)
        ax = axes[2] 
        if self.ela_dataset is not None:
            # Create a 2D slice of ELA data for visualization
            ela_2d = self.ela_dataset.ELA.sel(T0=10)  
            X, Y = np.meshgrid(ela_2d.mu.values, ela_2d.gamma.values)
            im = ax.contourf(X, Y, ela_2d.values.T, levels=20, cmap='viridis')
            ax.set_xlabel('Melt Factor (μ)')
            ax.set_ylabel('Lapse Rate (γ, °C/km)')
            ax.set_title('ELA Distribution (T₀ = 10)')
            plt.colorbar(im, ax=ax, label='ELA (m)')
        
        # Plot 4: ELA across parameter space (if available)
        ax = axes[3]
        if self.ela_dataset is not None:
            # Create a 2D slice of ELA data for visualization
            ela_2d = self.ela_dataset.ELA.mean(dim='T0')  # Average over T0
            X, Y = np.meshgrid(ela_2d.mu.values, ela_2d.gamma.values)
            im = ax.contourf(X, Y, ela_2d.values.T, levels=20, cmap='viridis')
            ax.set_xlabel('Melt Factor (μ)')
            ax.set_ylabel('Lapse Rate (γ, °C/km)')
            ax.set_title('ELA Distribution (averaged over T₀)')
            plt.colorbar(im, ax=ax, label='ELA (m)')
        
        for ax in axes[:2]:
            ax.set_xlabel('Elevation (m)')
            ax.set_ylabel('Mass Balance (m w.e./yr)')
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # Convention: highest elevations on the left
            
        plt.tight_layout()
    def plot_ela_path_integrals(self, figsize=(15, 8), target_ela=2500, P0=1000, temp_change=5):
        """
        Create ELA path integral visualization showing linear vs nonlinear scaling
        
        Parameters:
        -----------
        target_ela : float
            Target initial ELA (m) for all parameter combinations
        P0 : float  
            Winter accumulation (mm)
        temp_change : float
            Temperature change to analyze (°C)
        """
        
        P0_m = P0 / 1000  # Convert to meters
        
        # Define four parameter combinations that give the same initial ELA
        # Two with same gamma (linear scaling), two with same mu (nonlinear scaling)
        combinations = [
            {'gamma': 6.0, 'mu': 0.5, 'label': 'γ=6, μ=0.5', 'style': '-', 'color': 'blue'},
            {'gamma': 6.0, 'mu': 1.0, 'label': 'γ=6, μ=1.0', 'style': '--', 'color': 'blue'},
            {'gamma': 5.0, 'mu': 0.75, 'label': 'γ=5, μ=0.75', 'style': '-', 'color': 'red'},
            {'gamma': 7.0, 'mu': 0.75, 'label': 'γ=7, μ=0.75', 'style': '--', 'color': 'red'}
        ]
        
        # Calculate initial T0 for each combination to achieve target_ela
        for combo in combinations:
            gamma_m = combo['gamma'] / 1000  # Convert to °C/m
            # From ELA = T0/gamma - P0/(mu*gamma), solve for T0:
            # T0 = gamma * (ELA + P0/(mu*gamma)) = gamma*ELA + P0/mu
            combo['T0_initial'] = gamma_m * target_ela + P0_m / combo['mu']
        
        # Create figure with 3 panels
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
        
        # Main panel: Temperature vs ELA paths
        ax_main = fig.add_subplot(gs[:, 0])
        
        for combo in combinations:
            gamma_m = combo['gamma'] / 1000
            T0_range = np.linspace(combo['T0_initial'], 
                                 combo['T0_initial'] + temp_change, 50)
            
            # Calculate ELA for each temperature
            ela_path = T0_range / gamma_m - P0_m / (combo['mu'] * gamma_m)
            
            ax_main.plot(T0_range, ela_path, 
                        linestyle=combo['style'], 
                        color=combo['color'],
                        linewidth=2.5,
                        label=combo['label'])
            
            # Mark initial and final points
            ax_main.plot(combo['T0_initial'], target_ela, 'o', 
                        color=combo['color'], markersize=8)
            ax_main.plot(T0_range[-1], ela_path[-1], 's', 
                        color=combo['color'], markersize=8)
        
        ax_main.set_xlabel('Sea Level Temperature (T₀, °C)', fontsize=12)
        ax_main.set_ylabel('ELA (m)', fontsize=12)
        ax_main.set_title(f'ELA Paths for +{temp_change}°C Temperature Change\n' + 
                         'Same γ = Linear Scaling (blue), Same μ = Nonlinear Scaling (red→red)', 
                         fontsize=14, pad=20)
        ax_main.legend(loc='best')
        ax_main.grid(True, alpha=0.3)
        
        # Add annotation about the area under curves
        ax_main.text(0.02, 0.98, 
                    'Area under curves:\n• Same γ: scales linearly\n• Same μ: scales nonlinearly',
                    transform=ax_main.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel 2: Contours with constant gamma (showing linear scaling)
        ax_gamma = fig.add_subplot(gs[0, 1])
        gamma_fixed = 6.0
        gamma_m = gamma_fixed / 1000
        
        # Create temperature and mu ranges for contours
        T0_contour = np.linspace(10, 25, 30)
        mu_contour = np.linspace(0.3, 1.5, 30)
        T0_mesh, mu_mesh = np.meshgrid(T0_contour, mu_contour)
        
        # Calculate ELA for contours
        ela_contour = T0_mesh / gamma_m - P0_m / (mu_mesh * gamma_m)
        
        contour_levels = np.arange(1500, 4000, 100)
        cs1 = ax_gamma.contour(T0_mesh, mu_mesh, ela_contour, levels=contour_levels, colors='gray', alpha=0.6)
        ax_gamma.clabel(cs1, inline=True, fontsize=8, fmt='%d m')
        
        # Highlight the specific combinations used
        for combo in combinations:
            if combo['gamma'] == gamma_fixed:
                ax_gamma.plot(combo['T0_initial'], combo['mu'], 'o', 
                            color=combo['color'], markersize=10, markeredgecolor='black')
                ax_gamma.text(combo['T0_initial'], combo['mu'] + 0.05, combo['label'], 
                            ha='center', fontsize=9, fontweight='bold')
        
        ax_gamma.set_xlabel('T₀ (°C)')
        ax_gamma.set_ylabel('Melt Factor (μ)')
        ax_gamma.set_title(f'ELA Contours\nγ = {gamma_fixed}°C/km (constant)', fontsize=11)
        ax_gamma.grid(True, alpha=0.3)
        
        # Panel 3: Contours with constant mu (showing nonlinear scaling)
        ax_mu = fig.add_subplot(gs[1, 1])
        mu_fixed = 0.75
        
        # Create temperature and gamma ranges for contours
        gamma_contour = np.linspace(4, 10, 30)
        T0_mesh, gamma_mesh = np.meshgrid(T0_contour, gamma_contour)
        gamma_mesh_m = gamma_mesh / 1000
        
        # Calculate ELA for contours
        ela_contour = T0_mesh / gamma_mesh_m - P0_m / (mu_fixed * gamma_mesh_m)
        
        cs2 = ax_mu.contour(T0_mesh, gamma_mesh, ela_contour, levels=contour_levels, colors='gray', alpha=0.6)
        ax_mu.clabel(cs2, inline=True, fontsize=8, fmt='%d m')
        
        # Show reference combinations (interpolated to mu_fixed)
        for combo in combinations:
            # Calculate what T0 would be for this gamma with mu_fixed
            gamma_m = combo['gamma'] / 1000
            T0_equiv = gamma_m * target_ela + P0_m / mu_fixed
            ax_mu.plot(T0_equiv, combo['gamma'], 's', 
                      color=combo['color'], markersize=8, markeredgecolor='black', alpha=0.7)
        
        ax_mu.set_xlabel('T₀ (°C)')
        ax_mu.set_ylabel('Lapse Rate (γ, °C/km)')
        ax_mu.set_title(f'ELA Contours\nμ = {mu_fixed} (constant)', fontsize=11)
        ax_mu.grid(True, alpha=0.3)
        
        # Panel 4: Phase space diagram
        ax_phase = fig.add_subplot(gs[0, 2])
        
        # Show the four combinations in parameter space
        gammas = [combo['gamma'] for combo in combinations]
        mus = [combo['mu'] for combo in combinations]
        colors = [combo['color'] for combo in combinations]
        
        scatter = ax_phase.scatter(mus, gammas, c=colors, s=100, edgecolors='black', linewidths=2)
        
        # Add labels
        for i, combo in enumerate(combinations):
            ax_phase.annotate(f"γ={combo['gamma']}\nμ={combo['mu']}", 
                            (combo['mu'], combo['gamma']), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=9, ha='left')
        
        ax_phase.set_xlabel('Melt Factor (μ)')
        ax_phase.set_ylabel('Lapse Rate (γ, °C/km)')
        ax_phase.set_title('Parameter Space\n(Same Initial ELA)', fontsize=11)
        ax_phase.grid(True, alpha=0.3)
        
        # Panel 5: Area under curves comparison
        ax_area = fig.add_subplot(gs[1, 2])
        
        # Calculate areas under the ELA paths (numerical integration)
        areas = []
        labels = []
        colors_area = []
        
        for combo in combinations:
            gamma_m = combo['gamma'] / 1000
            T0_range = np.linspace(combo['T0_initial'], 
                                 combo['T0_initial'] + temp_change, 50)
            ela_path = T0_range / gamma_m - P0_m / (combo['mu'] * gamma_m)
            
            # Calculate area using trapezoidal rule
            area = np.trapz(ela_path, T0_range)
            areas.append(area)
            labels.append(f"γ={combo['gamma']}\nμ={combo['mu']}")
            colors_area.append(combo['color'])
        
        bars = ax_area.bar(range(len(areas)), areas, color=colors_area, alpha=0.7, edgecolor='black')
        ax_area.set_xticks(range(len(areas)))
        ax_area.set_xticklabels(labels, fontsize=9)
        ax_area.set_ylabel('Area under ELA path')
        ax_area.set_title('Path Integral\nComparison', fontsize=11)
        ax_area.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, area) in enumerate(zip(bars, areas)):
            ax_area.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(areas)*0.01,
                        f'{area:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_ela_sensitivity(self, figsize=(12, 10)):
        """Create comprehensive ELA sensitivity plots"""
        
        if self.ela_dataset is None:
            raise ValueError("Must run create_parameter_sweep first")
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: ELA vs T0 for different lapse rates
        ax = axes[0,0]
        ela_data = self.ela_dataset.ELA
        gamma_values = ela_data.gamma.values[::2]  # Every 2nd value
        colors = cm.viridis(np.linspace(0, 1, len(gamma_values)))
        for i, gamma in enumerate(gamma_values):
            ela_slice = ela_data.sel(gamma=gamma).mean(dim='mu')
            ax.plot(ela_data.T0, ela_slice, label=f'γ={gamma:.1f}°C/km', color=colors[i])
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('ELA (m)')
        ax.set_title('ELA Sensitivity: Temperature vs Lapse Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: ELA vs T0 for different melt factors
        ax = axes[0,1]
        mu_values = ela_data.mu.values[::5]  # Every 5th value
        colors = cm.plasma(np.linspace(0, 1, len(mu_values)))
        for i, mu in enumerate(mu_values):
            ela_slice = ela_data.sel(mu=mu, method='nearest').mean(dim='gamma')
            ax.plot(ela_data.T0, ela_slice, label=f'μ={mu:.2f}', color=colors[i])
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('ELA (m)')
        ax.set_title('ELA Sensitivity: Temperature vs Melt Factor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Hide the third panel in top row
        axes[0,2].set_visible(False)
        
        # Plot 4: 2D heatmap ELA vs mu and gamma
        ax = axes[1,0]
        ela_2d = ela_data.mean(dim='T0')
        X, Y = np.meshgrid(ela_2d.mu.values, ela_2d.gamma.values)
        im1 = ax.contourf(X, Y, ela_2d.values.T, levels=20, cmap='viridis')
        ax.set_xlabel('Melt Factor (μ)')
        ax.set_ylabel('Lapse Rate (γ, °C/km)')
        ax.set_title('ELA: μ vs γ (avg T₀)')
        plt.colorbar(im1, ax=ax, label='ELA (m)')
        
        # Plot 5: 2D heatmap ELA vs T0 and gamma
        ax = axes[1,1]
        ela_2d = ela_data.mean(dim='mu')
        X, Y = np.meshgrid(ela_2d.T0.values, ela_2d.gamma.values)
        im2 = ax.contourf(X, Y, ela_2d.values, levels=20, cmap='plasma')
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('Lapse Rate (γ, °C/km)')
        ax.set_title('ELA: T₀ vs γ (avg μ)')
        plt.colorbar(im2, ax=ax, label='ELA (m)')
        
        # Plot 6: 2D heatmap ELA vs T0 and mu
        ax = axes[1,2]
        ela_2d = ela_data.mean(dim='gamma')
        X, Y = np.meshgrid(ela_2d.T0.values, ela_2d.mu.values)
        im3 = ax.contourf(X, Y, ela_2d.values, levels=20, cmap='coolwarm')
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('Melt Factor (μ)')
        ax.set_title('ELA: T₀ vs μ (avg γ)')
        plt.colorbar(im3, ax=ax, label='ELA (m)')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_ela_surface(self, fixed_param='T0', fixed_value=None, figsize=(12, 8)):
        """Create 3D surface plot of ELA"""
        
        if self.ela_dataset is None:
            raise ValueError("Must run create_parameter_sweep first")
            
        from mpl_toolkits.mplot3d import Axes3D
        
        ela_data = self.ela_dataset.ELA
        
        # Select which parameter to fix
        if fixed_value is None:
            if fixed_param == 'T0':
                fixed_value = ela_data.T0.values[len(ela_data.T0)//2]
            elif fixed_param == 'mu':
                fixed_value = ela_data.mu.values[len(ela_data.mu)//2]
            elif fixed_param == 'gamma':
                fixed_value = ela_data.gamma.values[len(ela_data.gamma)//2]
            elif fixed_param == 'ELA':
                fixed_value = float(np.median(ela_data.values))  # Use median ELA value
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if fixed_param == 'T0':
            ela_slice = ela_data.sel(T0=fixed_value, method='nearest')
            X, Y = np.meshgrid(ela_slice.mu.values, ela_slice.gamma.values)
            Z = ela_slice.values.T
            ax.set_xlabel('Melt Factor (μ)')
            ax.set_ylabel('Lapse Rate (γ, °C/km)')
            ax.set_zlabel('ELA (m)')
            title = f'ELA Surface (T₀ = {fixed_value:.1f}°C)'
            
        elif fixed_param == 'mu':
            ela_slice = ela_data.sel(mu=fixed_value, method='nearest')
            X, Y = np.meshgrid(ela_slice.gamma.values, ela_slice.T0.values)
            Z = ela_slice.values.T
            ax.set_xlabel('Lapse Rate (γ, °C/km)')
            ax.set_ylabel('Sea Level Temperature (T₀, °C)')
            ax.set_zlabel('ELA (m)')
            title = f'ELA Surface (μ = {fixed_value:.2f})'
            
        elif fixed_param == 'gamma':
            ela_slice = ela_data.sel(gamma=fixed_value, method='nearest')
            X, Y = np.meshgrid(ela_slice.mu.values, ela_slice.T0.values)
            Z = ela_slice.values.T
            ax.set_xlabel('Melt Factor (μ)')
            ax.set_ylabel('Sea Level Temperature (T₀, °C)')
            ax.set_zlabel('ELA (m)')
            title = f'ELA Surface (γ = {fixed_value:.1f}°C/km)'
        
        elif fixed_param == 'ELA':
            from scipy.interpolate import interp1d, griddata
            
            # Arrays to store isosurface points
            mu_iso = []
            gamma_iso = []
            T0_iso = []
            
            # For each mu-gamma combination, interpolate to find T0 where ELA = fixed_value
            for i, mu_val in enumerate(ela_data.mu.values):
                for j, gamma_val in enumerate(ela_data.gamma.values):
                    # Get ELA values along T0 dimension
                    ela_slice = ela_data[i, j, :].values
                    T0_vals = ela_data.T0.values
                    
                    # Check if target ELA is within the range
                    if ela_slice.min() <= fixed_value <= ela_slice.max():
                        # Interpolate to find exact T0 for target ELA
                        try:
                            f = interp1d(ela_slice, T0_vals, kind='linear', bounds_error=False)
                            T0_interp = f(fixed_value)
                            
                            if not np.isnan(T0_interp):
                                mu_iso.append(mu_val)
                                gamma_iso.append(gamma_val)
                                T0_iso.append(T0_interp)
                        except:
                            continue
            
            if len(mu_iso) < 4:
                raise ValueError(f"Not enough points found for ELA = {fixed_value}m surface")
            
            # Convert to arrays
            mu_iso = np.array(mu_iso)
            gamma_iso = np.array(gamma_iso)
            T0_iso = np.array(T0_iso)
            
            # Create regular grid and interpolate
            mu_grid = np.linspace(mu_iso.min(), mu_iso.max(), 20)
            gamma_grid = np.linspace(gamma_iso.min(), gamma_iso.max(), 15)
            MU_grid, GAMMA_grid = np.meshgrid(mu_grid, gamma_grid)
            
            points = np.column_stack([mu_iso, gamma_iso])
            T0_grid = griddata(points, T0_iso, (MU_grid, GAMMA_grid), method='cubic', fill_value=np.nan)
            
            # Plot surface
            X = MU_grid
            Y = GAMMA_grid
            Z = T0_grid
            ax.set_xlabel('Melt Factor (μ)')
            ax.set_ylabel('Lapse Rate (γ, °C/km)')
            ax.set_zlabel('Sea Level Temperature (T₀, °C)')
            title = f'Parameter Space for ELA = {fixed_value:.0f}m'
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_title(title)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, label='ELA (m)')
        
        return fig

# Example usage and demonstration
if __name__ == "__main__":
    # Create analysis instance
    glacier_analysis = GlacierELAAnalysis()
    
    # Create parameter sweep
    print("Creating parameter sweep...")
    mb_data, ela_data = glacier_analysis.create_parameter_sweep(
        P0=1000  # 1000 mm winter accumulation (converted to m internally)
    )
    
    print(f"Mass balance dataset shape: {mb_data.mass_balance.shape}")
    print(f"ELA dataset shape: {ela_data.ELA.shape}")
    
    # Create visualizations
    print("Creating mass balance profiles...")
    fig1 = glacier_analysis.plot_mass_balance_profiles()
    plt.show()
    
    print("Creating ELA sensitivity analysis...")
    fig2 = glacier_analysis.plot_ela_sensitivity()
    plt.show()
    
    print("Creating 3D ELA surface...")
    fig3 = glacier_analysis.plot_3d_ela_surface()
    plt.show()
    
    fig3 = glacier_analysis.plot_3d_ela_surface(fixed_param="gamma")
    plt.show()
    
    fig3 = glacier_analysis.plot_3d_ela_surface(fixed_param="mu")
    plt.show()
    
    fig3 = glacier_analysis.plot_3d_ela_surface(fixed_param="ELA")
    plt.show()
    
    fig3 = glacier_analysis.plot_3d_ela_surface()
    plt.show()
    
    print("Creating 3D ELA surface...")
    fig4 = glacier_analysis.plot_ela_path_integrals()
    plt.show()
    
    # Print some statistics
    print("\nELA Statistics:")
    print(f"Min ELA: {ela_data.ELA.min().values:.0f} m")
    print(f"Max ELA: {ela_data.ELA.max().values:.0f} m") 
    print(f"Mean ELA: {ela_data.ELA.mean().values:.0f} m")
    print(f"Std ELA: {ela_data.ELA.std().values:.0f} m")