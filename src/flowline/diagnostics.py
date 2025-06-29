import numpy as np
import pandas as pd
import scipy as sci
import numba as nb
from functools import partial

def calc_ela(P0, T0, gamma, mu, h=None):
    # this seems to be accurate with elev mb feedback??
    if np.asarray(h).any():  # idk if this part is right
        T0 = T0 - h * gamma
    ela = T0 / gamma - P0 / (mu * gamma)
    return ela

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
    b = res.gwb / res.area
    diag.loc['b', 'mean'] = b[tslice].mean()
    diag.loc['b', 'std'] = b[tslice].std()
    diag.loc['b', 'mean_025'], diag.loc['b', 'mean_975'] = sci.stats.t.interval(
        0.95, df, loc=diag.loc['b', 'mean'], scale=diag.loc['b', 'std']
    )
    diag.loc['b', 'std_025'] = diag.loc['b', 'std_975'] = np.nan
    try:
        diag.loc['T', 'std'] = res.T[tslice].mean(axis=1).std()
        diag.loc['P', 'std'] = res.P[tslice].mean(axis=1).std()
    except:
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
    babl = np.array([res.b[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
    bacc = np.array([res.b[i, :j].mean() for i, j in enumerate(res.ela_idx[tslice])])
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
    w = res.w.reshape(1, -1).repeat(10000, 0)
    wabl = np.array([w[i, j[0] : j[1]].mean() for i, j in enumerate(zip(res.ela_idx[tslice], res.edge_idx[tslice]))])
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
    aar = np.array([w[i, 0:j].sum() * res.delx for i, j in enumerate(res.ela_idx[tslice])]) / res.area[tslice]
    diag.loc['aar', 'mean'] = aar.mean()
    diag.loc['aar', 'std'] = aar.std()
    return diag
