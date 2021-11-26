import numpy as np
import gstools as gs


def batch_generate_krige(nx, ny, lx, ly,
                         variance,
                         len_scale,
                         cond_pos,
                         cond_val,
                         angle,
                         n_samples,
                         seed):
    
    '''
    nx, ny, lx, ly: grid dicretization  (nx and ny) and length (lx and ly) in x and y directions  
    variance: variance value in exponential variagram (\sigma in equation 8)
    len_scale: length scale for the exponential variogram (l_1 and l_2 in equation 9)
    cond_pos: an array of positions where distribution values are known
    cond_val: an array of values corresponding to 'cond_pos' locations
    angle: rotation angle of the generated field
    n_samples: number of samples to be generated
    seed: random state value for reproducibility
    
    '''
    
    # generate variogram model
    model = gs.Exponential(dim=2, var=variance, len_scale=len_scale, angles=angle)
    
    # ordinary kriging
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)
    srf = gs.CondSRF(krige)
    
    # generate samples
    step_x, step_y = lx/(nx-1), ly/(ny-1)
    g_cols, g_rows = np.arange(0,lx+1,step_x), np.arange(0,ly+1,step_y) # x and y directions correspond to matrix col and row 
    ks = []
    for i in range(n_samples): 
        k_ = srf.structured([g_rows, g_cols], seed=seed+i)
        ks.append(k_)
    ks = np.array(ks)
    
    return ks