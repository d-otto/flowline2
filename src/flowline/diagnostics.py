import numpy as np
import pandas as pd
import scipy as sci
import numba as nb
from functools import partial

@nb.njit()
def calc_ela(P0, T0, gamma, mu, h=None):
    """
    Calculate Equilibrium Line Altitude
    
    Parameters:
    -----------
    P0 : float/array
        Winter accumulation (m w.e.)
    T0 : float/array  
        Melt-season temperature at sea level (°C)
    gamma : float/array
        Temperature lapse rate (°C/m)
    mu : float/array
        Melt factor (m/°C/yr)
    h : float/array, optional
        Elevation of glacier surface (m)
        
    Returns:
    --------
    ela : float/array
        Equilibrium Line Altitude (m)
    """
    
    # Adjust temperature for elevation if provided
    if h is not None:
        T_z = T0 - h * gamma
    else:
        T_z = T0
        
    # Calculate ELA (mu is in m/°C/yr, P0_m is in m/yr)
    ela = T_z / gamma - P0 / (mu * gamma)
    return ela


@nb.njit()
def calc_mass_balance(h, P0, T0, gamma, mu):
    """
    Calculate mass balance at given elevation
    
    Parameters:
    -----------
    h : float/array
        Elevation (m)
    P0 : float/array
        Winter accumulation (m w.e.)
    T0 : float/array
        Melt-season temperature at sea level (°C)
    gamma : float/array
        Temperature lapse rate (°C/m)
    mu : float/array
        Melt factor (m/°C/yr)
        
    Returns:
    --------
    mass_balance : float/array
        Annual mass balance (m w.e./yr)
    """
    T_z = T0 - h * gamma  # Temperature at elevation z
    
    # Simple mass balance model: accumulation - melt
    # Melt only occurs when temperature > 0
    melt = np.maximum(0, mu * T_z)  # mu is in m/°C/yr
    mass_balance = P0 - melt  # Both in m w.e./yr
    
    return mass_balance

def calc_Leq(A, w, bt, db, L=None):
    if np.ndim(w) != 0:
        w = np.mean(w)
    return A / w * -db / bt

@nb.njit
def calc_tau3(h, b, edge_idx, toe_idx, term_idx):
    '''
    toe_idx = idx from terminus to start the terminus zone
    term_idx = idx after the start of the zone to end the zone
    '''
    n = h.shape[0]
    tau = np.empty(n)
    for i in range(n):
        j0, j1 = edge_idx[i] - toe_idx - term_idx, edge_idx[i] - toe_idx
        tau[i] = -h[i, j0:j1].mean() / b[i, j0:j1].mean()
    return tau

def calc_diag(res, t=(None, None)):
    """Calculate diagnostic statistics for model results"""
    tslice = slice(t[0], t[1])

    diag = pd.DataFrame(dtype=float, columns=['mean', 'std', 'mean_025', 'mean_975', 'std_025', 'std_975'])
    df = len(res.edge)
    b = res.total_mass_balance / res.area
    diag.loc['b', 'mean'] = b[tslice].mean()
    diag.loc['b', 'std'] = b[tslice].std()
    diag.loc['b', 'mean_025'], diag.loc['b', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['b', 'mean'], scale=diag.loc['b', 'std']
    )
    diag.loc['b', 'std_025'] = diag.loc['b', 'std_975'] = np.nan
    try:
        diag.loc['T', 'std'] = res.T[tslice].std()
    except AttributeError:
        pass
    diag.loc['L', 'mean'] = res.edge[tslice].mean()
    diag.loc['L', 'std'] = res.edge[tslice].std()
    diag.loc['L', 'mean_025'], diag.loc['L', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['L', 'mean'], scale=diag.loc['L', 'std']
    )
    diag.loc['L', 'std_025'] = diag.loc['L', 'std_975'] = np.nan
    diag.loc['Hmax', 'mean'] = res.h[tslice].max(axis=1).mean()
    diag.loc['Hmax', 'std'] = res.h[tslice].max(axis=1).std()
    diag.loc['Hmax', 'mean_025'], diag.loc['Hmax', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['Hmax', 'mean'], scale=diag.loc['Hmax', 'std']
    )
    diag.loc['Area', 'mean'] = res.area[tslice].mean() / 1e6
    diag.loc['Area', 'std'] = res.area[tslice].std() / 1e6
    diag.loc['Area', 'mean_025'], diag.loc['Area', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['Area', 'mean'], scale=diag.loc['Area', 'std']
    )
    diag.loc['ELA', 'mean'] = res.ela[tslice].mean()
    diag.loc['ELA', 'std'] = res.ela[tslice].std()
    diag.loc['ELA', 'mean_025'], diag.loc['ELA', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['ELA', 'mean'], scale=diag.loc['ELA', 'std']
    )
    babl = np.array([res.b_profile[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
    bacc = np.array([res.b_profile[i, :j].mean() for i, j in enumerate(res.ela_idx[tslice])])
    diag.loc['babl', 'mean'] = np.nanmean(babl)
    diag.loc['bacc', 'mean'] = np.nanmean(bacc)
    diag.loc['babl', 'std'] = np.nanstd(babl)
    diag.loc['bacc', 'std'] = np.nanstd(bacc)
    diag.loc['babl', 'mean_025'], diag.loc['babl', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['babl', 'mean'], scale=diag.loc['babl', 'std']
    )
    diag.loc['bacc', 'mean_025'], diag.loc['bacc', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['bacc', 'mean'], scale=diag.loc['bacc', 'std']
    )
    Habl = np.array([res.h[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
    wabl = np.array([res.w[j[0] : j[1]].mean() for j in zip(res.ela_idx[tslice], res.edge_idx[tslice])])
    diag.loc['Habl', 'mean'] = Habl.mean()
    diag.loc['Habl', 'std'] = Habl.std()
    diag.loc['Habl', 'mean_025'], diag.loc['Habl', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['Habl', 'mean'], scale=diag.loc['Habl', 'std']
    )
    diag.loc['wabl', 'mean'] = wabl.mean()
    diag.loc['wabl', 'std'] = wabl.std()
    beta = res.area[tslice] / (Habl * wabl)
    diag.loc['beta', 'mean'] = beta.mean()
    diag.loc['beta', 'std'] = beta.std()
    aar = np.array([res.w[0:j].sum() * res.config.delx for j in res.ela_idx[tslice]]) / res.area[tslice]
    diag.loc['aar', 'mean'] = aar.mean()
    diag.loc['aar', 'std'] = aar.std()
    return diag


def log_model_setup(spinup_obj, output_file=None):
    """
    Print comprehensive model setup information from FlowlineSpinup object.
    
    Parameters
    ----------
    spinup_obj : FlowlineSpinup
        FlowlineSpinup object containing config, geometry, and forcing
    output_file : str, optional
        Path to file for writing output. If None, prints to console.
    """
    import sys
    from io import StringIO
    
    # Capture output
    if output_file:
        output = StringIO()
        file_handle = open(output_file, 'w')
    else:
        output = sys.stdout
        file_handle = None
    
    try:
        # Header
        print("=" * 80, file=output)
        print("FLOWLINE MODEL SETUP SUMMARY", file=output)
        print("=" * 80, file=output)
        
        # Model Configuration
        print("\n[MODEL CONFIGURATION]", file=output)
        config = spinup_obj.config
        print(f"  Physical Parameters:", file=output)
        print(f"    Ice density (rho):           {config.rho:.1f} kg/m³", file=output)
        print(f"    Gravity (g):                 {config.g:.2f} m/s²", file=output)
        print(f"    Deformation parameter (fd):  {config.fd:.2e} Pa⁻³ s⁻¹", file=output)
        print(f"    Sliding parameter (fs):      {config.fs:.2e} Pa⁻³ s⁻¹ m²", file=output)
        print(f"    Glen's flow law exponent:    {config.n}", file=output)
        print(f"    Sliding law exponent:        {config.k}", file=output)
        
        print(f"  Numerical Parameters:", file=output)
        print(f"    Grid spacing (delx):         {config.delx:.1f} m", file=output)
        print(f"    Time step (delt):            {config.delt:.6f} years", file=output)
        print(f"    Start time (ts):             {config.ts:.1f} years", file=output)
        print(f"    End time (tf):               {config.tf:.1f} years", file=output)
        print(f"    Simulation duration:         {config.tf - config.ts:.1f} years", file=output)
        print(f"    Minimum terminus thickness:  {config.min_thick:.1f} m", file=output)
        
        print(f"  Climate Parameters:", file=output)
        print(f"    Temperature lapse rate:      {config.gamma:.1e} °C/m", file=output)
        print(f"    Melt factor (mu):            {config.mu:.3f} m/°C/yr", file=output)
        print(f"    Height-mass balance feedback: {config.hmb}", file=output)
        
        # Geometry Information
        print("\n[GEOMETRY CONFIGURATION]", file=output)
        geometry = spinup_obj.geometry
        print(f"  Domain Characteristics:", file=output)
        print(f"    Domain extent:               {geometry.x_gr.min():.0f} - {geometry.x_gr.max():.0f} m", file=output)
        print(f"    Domain length:               {geometry.x_gr.max() - geometry.x_gr.min():.0f} m", file=output)
        print(f"    High-res geometry points:    {len(geometry.x_gr)}", file=output)
        
        print(f"  Bed Elevation Statistics:", file=output)
        print(f"    Minimum bed elevation:       {geometry.zb_gr.min():.0f} m", file=output)
        print(f"    Maximum bed elevation:       {geometry.zb_gr.max():.0f} m", file=output)
        print(f"    Mean bed elevation:          {geometry.zb_gr.mean():.0f} m", file=output)
        print(f"    Elevation range:             {geometry.zb_gr.max() - geometry.zb_gr.min():.0f} m", file=output)
        print(f"    Mean bed slope:              {np.gradient(geometry.zb_gr, geometry.x_gr).mean():.4f} m/m", file=output)
        
        print(f"  Width Characteristics:", file=output)
        print(f"    Minimum width:               {geometry.w_geom.min():.0f} m", file=output)
        print(f"    Maximum width:               {geometry.w_geom.max():.0f} m", file=output)
        print(f"    Mean width:                  {geometry.w_geom.mean():.0f} m", file=output)
        
        # Check if grid has been set up
        if hasattr(geometry, 'x') and geometry.x is not None:
            print(f"  Model Grid (after setup):", file=output)
            print(f"    Model grid points:           {len(geometry.x)}", file=output)
            print(f"    Actual grid spacing:         {geometry.x[1] - geometry.x[0]:.1f} m", file=output)
        
        # Initial thickness info if available
        if hasattr(geometry, 'h0') and geometry.h0 is not None:
            print(f"  Initial Ice Thickness:", file=output)
            print(f"    Maximum thickness:           {geometry.h0.max():.1f} m", file=output)
            print(f"    Mean thickness:              {geometry.h0.mean():.1f} m", file=output)
            print(f"    Initial glacier length:      {(geometry.h0 > 0).sum() * config.delx:.0f} m", file=output)
        
        # Forcing Information
        print("\n[FORCING CONFIGURATION]", file=output)
        forcing = spinup_obj.forcing
        forcing_type = type(forcing).__name__
        print(f"  Forcing Type: {forcing_type}", file=output)
        
        if forcing_type == "TemperaturePrecipitationForcing":
            print(f"  Climate Parameters:", file=output)
            print(f"    Reference temperature (T0):  {forcing.T0:.2f} °C", file=output)
            print(f"    Reference precipitation (P0): {forcing.P0:.3f} m w.e./yr", file=output)
            print(f"    Temperature lapse rate:       {forcing.gamma:.1e} °C/m", file=output)
            print(f"    Melt factor (mu):             {forcing.mu:.3f} m w.e./°C/yr", file=output)
            
            # Calculate estimated ELA
            try:
                estimated_ela = calc_ela(forcing.P0, forcing.T0, forcing.gamma, forcing.mu)
                print(f"    Estimated ELA:                {estimated_ela:.0f} m", file=output)
            except:
                print(f"    Estimated ELA:                Could not calculate", file=output)
            
            # Time series characteristics
            if hasattr(forcing, 'Tp') and len(forcing.Tp) > 1:
                print(f"  Time Series Characteristics:", file=output)
                print(f"    Temperature variability (std): {forcing.Tp.std():.3f} °C", file=output)
                print(f"    Precipitation variability (std): {forcing.Pp.std():.3f} m w.e./yr", file=output)
            
        elif forcing_type == "DirectMassBalanceForcing":
            print(f"  Mass Balance Parameters:", file=output)
            print(f"    Base mass balance (b0):       {forcing.b0:.3f} m/yr", file=output)
            if forcing.bp is not None:
                if np.isscalar(forcing.bp):
                    print(f"    Mass balance anomaly:         {forcing.bp:.3f} m/yr (constant)", file=output)
                else:
                    print(f"    Mass balance anomaly:         time series (std: {np.std(forcing.bp):.3f} m/yr)", file=output)
            
            if forcing.dbdz is not None:
                print(f"    Elevation gradient:           Yes (dbdz array length: {len(forcing.dbdz)})", file=output)
            if forcing.dbdx is not None:
                print(f"    Distance gradient:            Yes (dbdx array length: {len(forcing.dbdx)})", file=output)
        
        # Target Matching Information
        if spinup_obj.target_matching is not None:
            print("\n[TARGET MATCHING CONFIGURATION]", file=output)
            tm = spinup_obj.target_matching
            
            if 'targets' in tm:
                print(f"  Target Values:", file=output)
                for key, value in tm['targets'].items():
                    if 'length' in key:
                        print(f"    {key}:                    {value:.0f} m", file=output)
                    elif 'thickness' in key:
                        print(f"    {key}:                 {value:.1f} m", file=output)
                    elif 'volume' in key:
                        print(f"    {key}:                   {value:.2e} m³", file=output)
                    else:
                        print(f"    {key}:                        {value}", file=output)
            
            print(f"  Optimization Settings:", file=output)
            print(f"    Adjustment parameter:         {tm.get('adjustment_parameter', 'Not specified')}", file=output)
            print(f"    Cost function:                {tm.get('cost_function', 'length_only')}", file=output)
            print(f"    Steady-state detector:        {tm.get('steady_state_detector', 'volume_change_rate')}", file=output)
            print(f"    Tolerance:                    {tm.get('tolerance', 100)}", file=output)
            print(f"    Parameter bounds:             {tm.get('parameter_bounds', 'Not specified')}", file=output)
            print(f"    Maximum iterations:           {tm.get('max_iterations', 10)}", file=output)
            print(f"    Maximum simulation time:      {tm.get('max_simulation_time', 1000)} years", file=output)
        
        # Summary
        print("\n[SUMMARY]", file=output)
        domain_length = geometry.x_gr.max() - geometry.x_gr.min()
        if forcing_type == "TemperaturePrecipitationForcing":
            try:
                ela = calc_ela(forcing.P0, forcing.T0, forcing.gamma, forcing.mu)
                ela_position = (ela - geometry.zb_gr.min()) / (geometry.zb_gr.max() - geometry.zb_gr.min())
                print(f"  ELA relative position in domain: {ela_position:.2f} (0=bottom, 1=top)", file=output)
            except:
                pass
        
        total_time_steps = int((config.tf - config.ts) / config.delt)
        print(f"  Total time steps in simulation:  {total_time_steps:,}", file=output)
        if hasattr(geometry, 'x') and geometry.x is not None:
            print(f"  Total grid points:               {len(geometry.x)}", file=output)
            print(f"  Computational cost estimate:     {total_time_steps * len(geometry.x):,} grid-point-steps", file=output)
        
        print("=" * 80, file=output)
        
        # Write to file if requested
        if output_file and file_handle:
            file_handle.write(output.getvalue())
            
    finally:
        if file_handle:
            file_handle.close()
