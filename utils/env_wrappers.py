import gym
from stable_baselines3.common.type_aliases import GymStepReturn
import numpy as np
from gym import spaces
from collections import deque
from utils.coarse_grid_functions import get_accmap, fine_to_coarse_mapping, coarse_to_fine_mapping
from numpy import sum, mean
from scipy.stats import hmean
from copy import copy
from gym import spaces
from ressim_env import ResSimEnv_v0, ResSimEnv_v1


class StepReset(gym.Wrapper):
    def __init__(self, env: gym.Env, steps_max: int = 1):
        """
        'steps_max' no. of steps with reset

        """
        gym.Wrapper.__init__(self, env)
        self.steps_max = steps_max
        self.reset_reward = 0.0

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.reset_reward = 0.0
        for _ in range(self.steps_max):
            action = np.array([1]*self.env.action_space.shape[0])
            obs, rew, done, _ = self.env.step(action)
            self.reset_reward = self.reset_reward + rew
        return obs
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        if self.env.episode_step == (self.steps_max+1):
            reward = reward + self.reset_reward
            
        return state, reward, done, info
    
    
class StateCoarse(gym.Wrapper):
    def __init__(self, env: gym.Env, x_coords: np.ndarray, y_coords: np.ndarray, include_well_pr: bool = False):
        """
        'x_coords' : x coordinates of the state grid to be considered
        'y_coords' : y coordinates of the state grid to be considered
        'include_well_pr' : append well (injectors and producers) pressure to states (assume zero pressure in reset)
        """
        gym.Wrapper.__init__(self, env)
        self.include_well_pr = include_well_pr
        self.x_coords = x_coords
        self.y_coords = y_coords
        if self.include_well_pr:
            high = np.array([1e5]* (x_coords.shape[0] + env.n_inj + env.n_prod) )
        else:
            high = np.array([1e5]*x_coords.shape[0])
        obs_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        self.observation_space = obs_space

    def reset(self, **kwargs) -> np.ndarray:
        state = self.env.reset(**kwargs)
        state = state.reshape(self.env.grid.shape)
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = np.zeros(self.env.n_inj + self.env.n_prod)
            obs = np.append(obs, ps)              
        return np.array(obs)
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        state = state.reshape(self.env.grid.shape)
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = []
            p_scaled = np.interp(self.env.solverP.p, (self.env.solverP.p.min(), self.env.solverP.p.max()), (-1,1))
            for x,y in zip(self.env.i_x, self.env.i_y):
                ps.append(p_scaled[x,y])
            for x,y in zip(self.env.p_x, self.env.p_y):
                ps.append(p_scaled[x,y])
            obs = np.append(obs, ps)   
        return np.array(obs), reward, done, info
    
    
class BufferWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_steps: int):
        """
        n_steps: number of steps to stack in state
        
        """
        gym.Wrapper.__init__(self, env)
        self.state_queue = deque(maxlen=n_steps)
        self.n_steps = n_steps
        high_ = self.env.observation_space.high.repeat(n_steps, axis=0)
        low_ = self.env.observation_space.low.repeat(n_steps, axis=0)
        obs_space = spaces.Box(low=low_, high=high_, dtype=np.float64)
        self.observation_space = obs_space

    def reset(self, **kwargs) -> np.ndarray:
        state = self.env.reset(**kwargs)
        for _ in range(self.n_steps):
            self.state_queue.append(state)
        return np.array(self.state_queue).reshape(-1)
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        self.state_queue.append(state)
        return np.array(self.state_queue).reshape(-1), reward, done, info
    
    
class EnvCoarseWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coarse_nx: int, coarse_ny: int):
        """
        coarse_nx : new coarse value for nx
        coarse_ny : new coarse calue for ny
        
        """
        gym.Wrapper.__init__(self, env)
        self.fine_grid = copy(self.env.grid)
        self.fine_s = copy(self.env.s)
        self.fine_q = copy(self.env.q)
        self.env.fine_n_inj = copy(self.env.n_inj)
        self.env.fine_n_prod = copy(self.env.n_prod)
        self.env.fine_i_x, self.env.fine_i_y = copy(self.env.i_x), copy(self.env.i_y)
        self.env.fine_p_x, self.env.fine_p_y = copy(self.env.p_x), copy(self.env.p_y)
        self.accmap = get_accmap(self.fine_grid, coarse_nx, coarse_ny)
        
        # coarsen grid
        self.env.grid.nx = coarse_nx
        self.env.grid.ny = coarse_ny
        
        # corasen k
        k_coarse = []
        for k in self.env.k_list:
            k_coarse.append(fine_to_coarse_mapping(k, self.accmap, func=hmean))
        self.env.k_list = np.array(k_coarse)
        
        # coarsen phi
        phi_coarse = fine_to_coarse_mapping(self.env.phi, self.accmap, func=mean)
        self.env.phi = phi_coarse
        
        # coarsen q
        q_coarse = fine_to_coarse_mapping(self.env.q, self.accmap, func=sum)
        self.env.q_init = q_coarse# storing inital values for reset function
        self.env.q = q_coarse
        
        # coarse injectors and producers
        self.env.tol = 1e-5
        self.env.Q = np.sum(self.env.q[self.env.q>self.env.tol])                                # total flow across the field
        self.env.n_inj = self.env.q[self.env.q>self.env.tol].size                          # no of injectors
        self.env.i_x, self.env.i_y =  np.where(self.env.q>self.env.tol)[0], np.where(self.env.q>self.env.tol)[1]    # injector co-ordinates
        self.env.n_prod = self.env.q[self.env.q<-self.env.tol].size                         # no of producers
        self.env.p_x, self.env.p_y =  np.where(self.env.q<-self.env.tol)[0], np.where(self.env.q<-self.env.tol)[1]    # producer co-ordinates
        
        # coarsen s
        s_coarse = fine_to_coarse_mapping(self.env.s, self.accmap, func=mean)
        self.env.s = s_coarse
        self.env.s_load = s_coarse
        self.env.state = self.env.s_load.reshape(-1)
        
        # original oil in place
        self.env.ooip = self.env.grid.lx * self.env.grid.ly * self.env.phi[0,0] * (1 - self.env.s_wir-self.env.s_oir)
        
        # keep state and action dimension corresponding to fine grid
        # state
        high = np.array([1e5]*self.fine_s.reshape(-1).shape[0])
        self.env.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        # action
        self.env.action_space = spaces.Box(low=np.array([0.001]*(self.env.fine_n_prod+self.env.fine_n_inj), dtype=np.float64),
                                           high=np.array([1]*(self.env.fine_n_prod+self.env.fine_n_inj),dtype=np.float64),
                                           dtype=np.float64)
        
    def reset(self, **kwargs) -> np.ndarray:
        coarse_state = self.env.reset(**kwargs)
        fine_state = self.coarse_to_fine_state_mapping(coarse_state)
        return fine_state
    
    def step(self, action) -> GymStepReturn:
        coarse_action = self.fine_to_coarse_action_mapping(action)
        coarse_state, reward, done, info = self.env.step(coarse_action)
        fine_state = self.coarse_to_fine_state_mapping(coarse_state)
        return fine_state, reward, done, info
    
    def fine_to_coarse_action_mapping(self, fine_action):
        fine_action_grid = np.zeros_like(self.fine_q)
        if isinstance(self.env,ResSimEnv_v1):
            inj_flow = fine_action[:self.fine_n_inj]
            fine_action_grid[self.fine_q>self.env.tol] = inj_flow
            prod_flow = fine_action[self.fine_n_inj:]
        if isinstance(self.env, ResSimEnv_v0):
            prod_flow = fine_action
        fine_action_grid[self.fine_q<-self.env.tol] = prod_flow
        
        coarse_action_grid = fine_to_coarse_mapping(fine_action_grid, self.accmap, func=sum)
        if isinstance(self.env,ResSimEnv_v1):
            inj_flow_coarse = coarse_action_grid[self.env.q>self.env.tol]
        if isinstance(self.env,ResSimEnv_v0):
            inj_flow_coarse = []
        prod_flow_coarse = coarse_action_grid[self.env.q<-self.env.tol]
        
        coarse_action = np.concatenate((inj_flow_coarse, prod_flow_coarse))
        
        return coarse_action
    
    def coarse_to_fine_state_mapping(self, coarse_state):
        coarse_s = coarse_state.reshape(self.env.grid.shape)
        fine_s = coarse_to_fine_mapping(coarse_s, self.accmap)
        fine_state = fine_s.reshape(-1)
        return fine_state
        
    
class StateCoarseMultiGrid(gym.Wrapper):
    def __init__(self, env: gym.Env, x_coords: np.ndarray, y_coords: np.ndarray, include_well_pr: bool = False):
        """
        'x_coords' : x coordinates of the state grid to be considered
        'y_coords' : y coordinates of the state grid to be considered
        'include_well_pr' : append well (injectors and producers) pressure to states (assume zero pressure in reset)
        """
        gym.Wrapper.__init__(self, env)
        self.include_well_pr = include_well_pr
        self.x_coords = x_coords
        self.y_coords = y_coords
        if self.include_well_pr:
            high = np.array([1e5]*(x_coords.shape[0] + env.fine_n_inj + env.fine_n_prod) )
        else:
            high = np.array([1e5]*x_coords.shape[0])
        obs_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        self.observation_space = obs_space

    def reset(self, **kwargs) -> np.ndarray:
        state = self.env.reset(**kwargs)
        state = state.reshape(self.env.fine_grid.shape)
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = np.zeros(self.env.fine_n_inj + self.env.fine_n_prod)
            obs = np.append(obs, ps)              
        return np.array(obs)
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        state = state.reshape(self.env.fine_grid.shape)
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = []
            p_scaled = np.interp(self.env.solverP.p, (self.env.solverP.p.min(), self.env.solverP.p.max()), (-1,1))
            p_scaled = coarse_to_fine_mapping(p_scaled, self.env.accmap)
            for x,y in zip(self.env.fine_i_x, self.env.fine_i_y):
                ps.append(p_scaled[x,y])
            for x,y in zip(self.env.fine_p_x, self.env.fine_p_y):
                ps.append(p_scaled[x,y])
            obs = np.append(obs, ps)   
        return np.array(obs), reward, done, info
    