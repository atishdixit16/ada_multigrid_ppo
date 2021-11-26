import numpy as np
from time import time
import os
from typing import Callable
from copy import copy, deepcopy
import gym

from utils.env_wrappers import EnvCoarseWrapper, StateCoarseMultiGrid
from utils.custom_eval_callback import CustomEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.plot_functions import plot_learning


def env_wrappers_multigrid(env, x_coords, y_coords, coarse_nx, coarse_ny, seed ):
    env_ = deepcopy(env)
    env_ = EnvCoarseWrapper(env_, coarse_nx=coarse_nx, coarse_ny=coarse_ny)
    env_ = StateCoarseMultiGrid(env_, x_coords, y_coords, include_well_pr=True)
    env_.seed(seed)
    return env_

def make_env(env, rank: int, seed: int) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env_ = env
        env_.seed(seed + rank)
        return env_
    return _init


def generate_beta_environement(env, beta, prods_loc_x, prods_loc_y, seed):
    env_ = env_wrappers_multigrid(env, 
                                  x_coords = prods_loc_x, 
                                  y_coords = prods_loc_y, 
                                  coarse_nx = int(env.grid.nx*beta), 
                                  coarse_ny = int(env.grid.ny*beta), 
                                  seed = seed )
    return env_

def parallalize_env(env, num_actor, seed):
    return SubprocVecEnv([make_env(env, i, seed) for i in range(num_actor)])  

def is_converged(r_array, 
                 n = 5, 
                 delta_pcent = 2):
    if r_array.shape[0] < n:
        return False
    r_grad = r_array[1:] - r_array[:-1]                # gradient
    r_grad_per = 100 * np.abs(r_grad) / r_array[:-1]   # convert into percent format
    delta_max = np.max(r_grad_per[-n:])
    return delta_max <= delta_pcent 


def multigrid_framework(env_train, 
                        generate_model,
                        generate_callback,
                        delta_pcent,
                        n, 
                        grid_fidelity_factor_array, 
                        episode_limit_array, 
                        log_dir,
                        seed):
    
    # generate and initialize the model using 'generate_model' function
    model = generate_model(env_train, seed)    
    
    # define log directory for the seed
    log_dir_seed = log_dir+'/seed_'+str(seed)
    os.makedirs(log_dir_seed, exist_ok=True)
    
    # generate callback to record iteration returns using 'generate_callback' function
    eval_freq = model.n_steps
    callback = generate_callback(env_train, str(log_dir_seed)+'/best_model_train', str(log_dir_seed)+'/results_train', eval_freq)
    
    total_episode_rollouts = 0
    
    # define episodes per iteration, (term 'E' in paper)
    num_actor = model.n_envs
    episodes_per_iteration = int((model.n_steps*num_actor)/env_train.terminal_step)
    
    for i, beta in enumerate(grid_fidelity_factor_array):
        
        # generate environement with \beta_i grid fidelity factor
        env = generate_beta_environement(env_train, beta, env_train.p_x, env_train.p_y, seed)
        
        # update environements in RL model and callbacks with this environment
        callback.set_eval_env(env)
        model.env = parallalize_env(env, num_actor, seed)
        
        dummy = parallalize_env(env, num_actor, seed)
        
        # policy  iterations
        print(f'seed {seed}: grid fidelity factor {beta} learning ..')
        nx, ny = model.get_env().get_attr('grid')[0].nx, model.get_env().get_attr('grid')[0].ny
        print(f'environement grid size (nx x ny ): {nx} x {ny}')
        for iter in range(0, episode_limit_array[i], episodes_per_iteration):
            
            # learn model with 'episodes_per_iteration' episodes
            before = time()            
            model.learn(total_timesteps= env.terminal_step*episodes_per_iteration, callback=callback)
            print(f'policy iteration runtime: {round(time()-before)} seconds')
            
            # record policy return array (\textbf{r} in the paper)
            with np.load(log_dir_seed+'/results_train/evaluations.npz') as data:
                r = data['results']
            
            # print output of total episode rollouts
            total_episode_rollouts = total_episode_rollouts + episodes_per_iteration
            print(f'\nTotal episode rollouts: {total_episode_rollouts}\n')
                
            # check convergence with parameters: \textbf{r}, n and \delta
            converge_flag = is_converged(r, n, delta_pcent)
            if converge_flag:
                break
                
    # monitor plots
    fig = plot_learning(log_dir_seed, case='train', multigrid=True)
    fig.savefig(log_dir_seed+'/learn_train.png')
    
    return model
                