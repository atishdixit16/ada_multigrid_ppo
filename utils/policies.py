import numpy as np
from scipy.interpolate import griddata


def five_spot_policy(env, param=0.5):
    
    '''
    rule based policy for five spot case
    producer flow is proportional to oil available nearby that producer
    
    parameters:
    
    param: % additional producer flow at producer with highest flow
    
    '''
    
    state = 1.0 - env.s_load  # oil saturation
    sum_ = np.sum(state)
    
    action = np.array([1.0]*env.action_space.shape[0])
    action[0] = np.sum( state[ :int(env.grid.nx/2), :int(env.grid.ny/2) ] ) / sum_
    action[1] = np.sum( state[ :int(env.grid.nx/2), int(env.grid.ny/2): ] ) / sum_
    action[2] = np.sum( state[ int(env.grid.nx/2):, :int(env.grid.ny/2) ] ) / sum_
    action[3] = np.sum( state[ int(env.grid.nx/2):, int(env.grid.ny/2): ] ) / sum_
    
    action[action.argmax()] = action.max()*(1+param)
    
    return action

def thirteen_spot_policy(env, param=None):
    
    state = 1.0 - env.s_load  # oil saturation
    action = np.array([1.0]*env.action_space.shape[0])
    
    # injector actions
    for i,(x,y) in enumerate( zip(env.i_x, env.i_y) ):
        x_l, x_r = np.max((x-15,0)), np.min((x+15,env.grid.nx))
        y_d, y_u = np.max((y-15,0)), np.min((y+15,env.grid.ny))
        action[i] = np.sum(state[x_l:x_r, y_d:y_u])
    action[:env.n_inj] = action[:env.n_inj]/np.sum(action[:env.n_inj])
    
    # producer actions
    for i,(x,y) in enumerate( zip(env.p_x, env.p_y) ):
        x_l, x_r = np.max((x-15,0)), np.min((x+15,env.grid.nx))
        y_d, y_u = np.max((y-15,0)), np.min((y+15,env.grid.ny))
        action[i+env.n_inj] = np.sum(state[x_l:x_r, y_d:y_u])
    action[env.n_inj:] = action[env.n_inj:]/np.sum(action[env.n_inj:])
    
    return action
    
    
            


def channel_case_policy(env, param=1.6):
    
    '''
    rule based policy for channel case
    injector flow near high permeability is restricted towards beginning of episode
    producer flow near high permeability is restricted towards the end of the episode
    
    parameters
    
    param: % additional width of high permeability region in which well control takes place
    
    '''
    
    k_p, k_i = env.k[:,-1], env.k[:,0]
    a_p, a_i = np.array( [float(k==k_p.min()) for k in k_p] ), np.array( [float(k==k_i.min()) for k in k_i] )
    
    grid = np.linspace(0,1,env.grid.ny)
    grid_i = np.linspace(0,1,env.n_inj)
    grid_p = np.linspace(0,1,env.n_prod)
    
    action_p = griddata(grid,a_p,grid_p, method='nearest')
    action_i = griddata(grid,a_i,grid_i, method='nearest')
    
    inj_opening = env.episode_step/env.terminal_step
    prod_opening = (env.terminal_step - 1 - env.episode_step)/env.terminal_step
    
    inj_min, inj_max = grid_i[np.where(action_i==0)].min(), grid_i[np.where(action_i==0)].max()
    prod_min, prod_max = grid_p[np.where(action_p==0)].min(), grid_p[np.where(action_p==0)].max()
    inj_min = inj_min - inj_min*(param/2)
    inj_max = inj_max + inj_max*(param/2)
    prod_min = prod_min - prod_min*(param/2)
    prod_max = prod_max + prod_max*(param/2)
    
    action_i [ np.intersect1d( np.where(grid_i >= inj_min ) , np.where( grid_i <= inj_max ) ) ] = 0
    action_p [ np.intersect1d( np.where(grid_p >= prod_min ) , np.where( grid_p <= prod_max ) ) ] = 0
    
    action_p[action_p==0] = prod_opening
    action_i[action_i==0] = inj_opening    
    
    action_array = np.append(action_i, action_p)
    action_array[action_array==0] = 1e-5
    
    return action_array
    
    