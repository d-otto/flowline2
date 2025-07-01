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
    def plot_ela_warming_response(self, target_ela=2500, P0=1000, temp_change=1.0, n_steps=20, figsize=(12, 8)):
        """
        Analyzes and visualizes glacier ELA and mass balance response to a uniform temperature change.

        Parameters:
        -----------
        target_ela : float
            Target initial ELA (m) for all parameter combinations.
        P0 : float  
            Winter accumulation (mm).
        temp_change : float
            Total temperature change to analyze (°C).
        n_steps : int
            Number of steps to model the temperature change.
        """
        if self.dataset is None:
            raise ValueError("Must run create_parameter_sweep first")

        P0_m = P0 / 1000  # Convert to meters
        elevations = self.dataset.elevation.values

        combinations = [
            {'gamma': 6.0, 'mu': 0.5, 'label': 'γ=6, μ=0.5', 'style': '-', 'color': 'blue'},
            {'gamma': 6.0, 'mu': 1.0, 'label': 'γ=6, μ=1.0', 'style': '--', 'color': 'dodgerblue'},
            {'gamma': 5.0, 'mu': 0.75, 'label': 'γ=5, μ=0.75', 'style': '-', 'color': 'red'},
            {'gamma': 7.0, 'mu': 0.75, 'label': 'γ=7, μ=0.75', 'style': '--', 'color': 'indianred'}
        ]

        for combo in combinations:
            gamma_m = combo['gamma'] / 1000
            combo['T0_initial'] = gamma_m * target_ela + P0_m / combo['mu']

        # Create figures
        fig_main, ax_main = plt.subplots(figsize=figsize)
        fig_profiles, axes_profiles = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        ax_const_gamma, ax_const_mu = axes_profiles

        for combo in combinations:
            gamma_m = combo['gamma'] / 1000
            T0_initial = combo['T0_initial']
            
            delta_T_path = np.linspace(0, temp_change, n_steps)
            ela_path = []
            delta_B_path = []

            b_initial = self.calc_mass_balance(elevations, P0, T0_initial, combo['gamma'], combo['mu'])

            for delta_T in delta_T_path:
                T0_current = T0_initial + delta_T
                ela_current = self.calc_ela(P0, T0_current, combo['gamma'], combo['mu'])
                ela_path.append(ela_current)

                b_current = self.calc_mass_balance(elevations, P0, T0_current, combo['gamma'], combo['mu'])
                delta_B = np.trapz(b_current - b_initial, elevations)
                delta_B_path.append(delta_B)

            ax_main.plot(delta_B_path, ela_path, label=combo['label'], color=combo['color'], linestyle=combo['style'], linewidth=2.5)
            
            # Plot initial and final points
            ax_main.plot(delta_B_path[0], ela_path[0], 'o', color=combo['color'], markersize=8)
            ax_main.plot(delta_B_path[-1], ela_path[-1], 's', color=combo['color'], markersize=8)

            # Plot mass balance profiles
            b_final = self.calc_mass_balance(elevations, P0, T0_initial + temp_change, combo['gamma'], combo['mu'])
            if combo['mu'] == 0.75: # Constant mu cases
                ax_const_mu.plot(b_initial, elevations, color=combo['color'], linestyle=':', label=f"Initial {combo['label']}")
                ax_const_mu.plot(b_final, elevations, color=combo['color'], linestyle=combo['style'], label=f"Final {combo['label']}")
                ax_const_mu.fill_betweenx(elevations, b_initial, b_final, color=combo['color'], alpha=0.1)
            else: # Constant gamma cases
                ax_const_gamma.plot(b_initial, elevations, color=combo['color'], linestyle=':', label=f"Initial {combo['label']}")
                ax_const_gamma.plot(b_final, elevations, color=combo['color'], linestyle=combo['style'], label=f"Final {combo['label']}")
                ax_const_gamma.fill_betweenx(elevations, b_initial, b_final, color=combo['color'], alpha=0.2)
        
        # Finalize main plot
        ax_main.set_xlabel('Change in Integrated Mass Balance Profile (m²/yr)', fontsize=12)
        ax_main.set_ylabel('ELA (m)', fontsize=12)
        ax_main.set_title(f'ELA Response to +{temp_change}°C Warming', fontsize=14)
        ax_main.legend()
        ax_main.grid(True, alpha=0.4)

        # Finalize profile plots
        for ax in axes_profiles:
            ax.axvline(0, color='black', linestyle='--', alpha=0.7)
            ax.set_xlabel('Mass Balance (m w.e./yr)')
            ax.set_ylabel('Elevation (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        ax_const_gamma.set_title('Mass Balance Profiles (Constant γ)')
        ax_const_mu.set_title('Mass Balance Profiles (Constant μ)')
        fig_profiles.tight_layout()

        return fig_main, fig_profiles
    
    def plot_ela_sensitivity(self, figsize=(12, 12)):
        """Create comprehensive ELA sensitivity plots"""
        
        if self.ela_dataset is None:
            raise ValueError("Must run create_parameter_sweep first")
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ela_data = self.ela_dataset.ELA
        P0_m = self.ela_dataset.P0.item() / 1000

        # Plot 1: 2D heatmap ELA vs mu and gamma
        ax = axes[0,0]
        ela_2d = ela_data.mean(dim='T0')
        X, Y = np.meshgrid(ela_2d.mu.values, ela_2d.gamma.values)
        im1 = ax.contourf(X, Y, ela_2d.values.T, levels=20, cmap='viridis')
        ax.set_xlabel('Melt Factor (μ)')
        ax.set_ylabel('Lapse Rate (γ, °C/km)')
        ax.set_title('ELA: μ vs γ (avg T₀)')
        plt.colorbar(im1, ax=ax, label='ELA (m)')
        
        # Plot 2: 2D heatmap ELA vs T0 and gamma
        ax = axes[0,1]
        ela_2d = ela_data.mean(dim='mu')
        X, Y = np.meshgrid(ela_2d.T0.values, ela_2d.gamma.values)
        im2 = ax.contourf(X, Y, ela_2d.values, levels=20, cmap='plasma')
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('Lapse Rate (γ, °C/km)')
        ax.set_title('ELA: T₀ vs γ (avg μ)')
        plt.colorbar(im2, ax=ax, label='ELA (m)')
        
        # Plot 3: 2D heatmap ELA vs T0 and mu
        ax = axes[1,0]
        ela_2d = ela_data.mean(dim='gamma')
        X, Y = np.meshgrid(ela_2d.T0.values, ela_2d.mu.values)
        im3 = ax.contourf(X, Y, ela_2d.values, levels=20, cmap='coolwarm')
        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('Melt Factor (μ)')
        ax.set_title('ELA: T₀ vs μ (avg γ)')
        plt.colorbar(im3, ax=ax, label='ELA (m)')

        # Plot 4: Overlapping Contours of mu and gamma
        ax = axes[1,1]
        T0_vals = np.linspace(ela_data.T0.min().item(), ela_data.T0.max().item(), 50)
        ELA_vals = np.linspace(ela_data.min().item(), ela_data.max().item(), 50)
        T0_mesh, ELA_mesh = np.meshgrid(T0_vals, ELA_vals)

        # Plot contours of constant gamma
        gamma_contour_vals = np.arange(4, 11, 1.0)
        for gamma in gamma_contour_vals:
            gamma_m = gamma / 1000
            with np.errstate(divide='ignore', invalid='ignore'):
                mu_iso = P0_m / (T0_mesh - ELA_mesh * gamma_m)
            mu_iso[mu_iso < 0] = np.nan
            cs_gamma = ax.contour(T0_mesh, ELA_mesh, mu_iso, levels=[0.5, 1.0, 1.5], linestyles='--', cmap='winter')
            ax.clabel(cs_gamma, inline=True, fontsize=8, fmt=f'γ={gamma:.0f}, μ=%1.1f')
        
        # Plot contours of constant mu
        mu_contour_vals = np.arange(0.4, 1.5, 0.2)
        for mu in mu_contour_vals:
            with np.errstate(divide='ignore', invalid='ignore'):
                gamma_iso_m = (T0_mesh - P0_m/mu) / ELA_mesh
            gamma_iso = gamma_iso_m * 1000
            gamma_iso[gamma_iso < 0] = np.nan
            cs_mu = ax.contour(T0_mesh, ELA_mesh, gamma_iso, levels=[5, 7, 9], linestyles='-', cmap='autumn')
            ax.clabel(cs_mu, inline=True, fontsize=8, fmt=f'μ={mu:.1f}, γ=%1.0f')

        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('ELA (m)')
        ax.set_title('ELA vs T₀ (Contours of μ and γ)')
        ax.set_ylim(1000, 4500)
        ax.grid(True, alpha=0.3)
        
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
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            plt.colorbar(surf, ax=ax, shrink=0.5, label='ELA (m)')
            
        elif fixed_param == 'ELA':
            from scipy.interpolate import griddata
            
            # Arrays to store isosurface points
            mu_iso = []
            gamma_iso = []
            T0_iso = []
            
            # For each mu-gamma combination, interpolate to find T0 where ELA = fixed_value
            for i, mu_val in enumerate(ela_data.mu.values):
                for j, gamma_val in enumerate(ela_data.gamma.values):
                    # Get ELA values along T0 dimension
                    ela_slice = ela_data.sel(mu=mu_val, gamma=gamma_val).values
                    T0_vals = ela_data.T0.values
                    
                    # Check if target ELA is within the range
                    if np.nanmin(ela_slice) <= fixed_value <= np.nanmax(ela_slice):
                        # Interpolate to find exact T0 for target ELA
                        T0_interp = np.interp(fixed_value, ela_slice, T0_vals)
                        
                        if not np.isnan(T0_interp):
                            mu_iso.append(mu_val)
                            gamma_iso.append(gamma_val)
                            T0_iso.append(T0_interp)
            
            if len(mu_iso) < 4:
                raise ValueError(f"Not enough points found for ELA = {fixed_value}m surface")
            
            # Convert to arrays
            mu_iso = np.array(mu_iso)
            gamma_iso = np.array(gamma_iso)
            T0_iso = np.array(T0_iso)
            
            # Create regular grid and interpolate
            mu_grid = np.linspace(mu_iso.min(), mu_iso.max(), 30)
            gamma_grid = np.linspace(gamma_iso.min(), gamma_iso.max(), 30)
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
            plt.colorbar(surf, ax=ax, shrink=0.5, label='T₀ (°C)')
        
        ax.set_title(title)
        return fig
    
    def plot_combined_3d_isolines(self, figsize=(12, 9)):
        """
        Creates a 3D plot with ELA isolines on planes of constant mu and gamma.
        """
        if self.ela_dataset is None:
            raise ValueError("Must run create_parameter_sweep first")

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ela_data = self.ela_dataset.ELA

        # --- Plane of constant gamma ---
        gamma_fixed = 6.0
        ela_slice_gamma = ela_data.sel(gamma=gamma_fixed, method='nearest')
        T0_mesh, mu_mesh = np.meshgrid(ela_slice_gamma.T0.values, ela_slice_gamma.mu.values)
        Z = ela_slice_gamma.values.T
        
        cs1 = ax.contour(T0_mesh, mu_mesh, Z, zdir='z', offset=gamma_fixed, 
                         levels=np.arange(1000, 5000, 500), cmap='winter')

        # --- Plane of constant mu ---
        mu_fixed = 0.75
        ela_slice_mu = ela_data.sel(mu=mu_fixed, method='nearest')
        T0_mesh, gamma_mesh = np.meshgrid(ela_slice_mu.T0.values, ela_slice_mu.gamma.values)
        Z = ela_slice_mu.values
        
        cs2 = ax.contour(T0_mesh, Z, gamma_mesh, zdir='y', offset=mu_fixed, 
                         levels=np.arange(1000, 5000, 500), cmap='autumn')

        ax.set_xlabel('Sea Level Temperature (T₀, °C)')
        ax.set_ylabel('Melt Factor (μ)')
        ax.set_zlabel('Lapse Rate (γ, °C/km)')
        ax.set_title('ELA Isolines in Parameter Space')
        
        ax.set_xlim(ela_data.T0.min(), ela_data.T0.max())
        ax.set_ylim(ela_data.mu.min(), ela_data.mu.max())
        ax.set_zlim(ela_data.gamma.min(), ela_data.gamma.max())

        # Dummy lines for legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', lw=2, label=f'Constant γ = {gamma_fixed}°C/km'),
                           Line2D([0], [0], color='red', lw=2, label=f'Constant μ = {mu_fixed}')]
        ax.legend(handles=legend_elements)

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
    
    print("Creating 3D ELA surface (fixed T0)...")
    fig3a = glacier_analysis.plot_3d_ela_surface(fixed_param="T0")
    plt.show()

    print("Creating combined 3D ELA isolines...")
    fig3b = glacier_analysis.plot_combined_3d_isolines()
    plt.show()
    
    print("Creating 3D ELA surface (fixed ELA)...")
    fig3c = glacier_analysis.plot_3d_ela_surface(fixed_param="ELA")
    plt.show()
    
    print("Creating ELA warming response analysis...")
    fig4_main, fig4_profiles = glacier_analysis.plot_ela_warming_response()
    plt.show()
    
    # Print some statistics
    print("\nELA Statistics:")
    print(f"Min ELA: {ela_data.ELA.min().values:.0f} m")
    print(f"Max ELA: {ela_data.ELA.max().values:.0f} m") 
    print(f"Mean ELA: {ela_data.ELA.mean().values:.0f} m")
    print(f"Std ELA: {ela_data.ELA.std().values:.0f} m")
