import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import functools
from collections import deque

from model.ressim import SaturationEquation, PressureEquation
from model.utils import linear_mobility, quadratic_mobility, lamb_fn, f_fn, df_fn

'''
ResSimEnv_v0: action space consists of producer controls only
ResSimEnv_v1: action space consists of all wells controls
ResSimEnv_v2: variation of ResSimEnv_v0 (frequency of pr eqn steps)
ResSimEnv_v3: variation of ResSimEnv_v1 (frequency of pr eqn steps)

'''


class ResSimEnv_v0():

    def __init__(self,
                 grid, k, phi, s_wir, s_oir, # domain properties
                 mu_w, mu_o, mobility, # fluid properties
                 dt, nstep, terminal_step, # timesteps
                 q, s): # initial conditions

        # domain properties
        self.grid=grid
        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        self.k_list = k
        self.phi = phi
        self.s_wir = s_wir
        self.s_oir = s_oir

        # fluid properties
        self.mu_w = mu_w
        self.mu_o = mu_o
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'
        self.mobility = mobility

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.q_init = q.copy() # storing inital values for reset function
        self.q = q
        self.s = s

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # Model function (mobility and fractional flow function)
        if mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

        # RL parameters
        self.metadata = {'render.modes': []} # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.reward_range = (0.0, 1.0)       # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.spec = None                     # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py

        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.tol = 1e-5
        self.Q = np.sum(self.q[q>self.tol])                                # total flow across the field
        self.n_inj = 0 #self.q[self.q>self.tol].size                          # no of injectors
        self.i_x, self.i_y =  np.where(q>self.tol)[0], np.where(q>self.tol)[1]    # injector co-ordinates
        self.n_prod = self.q[self.q<-self.tol].size                         # no of producers
        self.p_x, self.p_y =  np.where(q<-self.tol)[0], np.where(q<-self.tol)[1]    # producer co-ordinates
        self.action_space = spaces.Box(low=np.array([0.001]*(self.n_prod), dtype=np.float64), 
                                       high=np.array([1]*(self.n_prod), dtype=np.float64), 
                                       dtype=np.float64)

        # for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping_cont(self, action):
        prod_flow = action / np.sum(action)
        prod_flow = -self.Q * prod_flow
        assert np.sum(prod_flow)<0, 'Invalid action: zero producer flow'+str(np.sum(prod_flow))
        q = self.q
        for x,y,i in zip(self.p_x, self.p_y, range(self.n_prod)):
            q[x,y] = prod_flow[i]
        if np.abs(np.sum(q)) < self.tol:
            q[3,3] = q[3,3] - np.sum(q) # to adjust unbalanced source term in arbitary location in the field due to precision error
        return q

    
    def sim_step(self, q):
        self.q = q
        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        oil_pr = 0.0
        # solve pressure
        self.solverP.s = self.solverS.s
        self.solverP.step()
        self.solverS.v = self.solverP.v
#         # cfl number
#         v_max = np.max([np.amax(self.solverP.v['x']), np.amax(self.solverP.v['y'])])
#         f_max = np.max([self.df_fn(s_) for s_ in np.arange(0,1.1,0.1) ])
#         cfl = (v_max*f_max*self.dt)/(self.phi[0,0]*self.grid.vol)
#         print(f'cfl: {cfl}')
        for _ in range(self.nstep):
            # solve saturation
            self.solverS.step(self.dt)
            self.s_load = self.solverS.s
            oil_pr = oil_pr + -np.sum( self.q[self.q<0] * ( 1- self.f_fn(self.s_load[self.q<0]) ) )*self.dt

        # state
        self.state = self.s_load.reshape(-1)

        #reward
        reward = oil_pr / self.ooip # recovery rate
#         reward = reward*100 # in percentage

        # done
        self.episode_step += 1
        if self.episode_step >= self.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}



    def step(self, action):
        q = self.action_to_q_mapping_cont(action)
        state, reward, done, info = self.sim_step(q)
        return state, reward, done, info

    def set_k(self, k):
        self.k_list = k
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        
    def set_observation_space(self, observation_space):
        self.observation_space = observation_space

    def reset(self):
        self.q = self.q_init
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        self.episode_step = 0
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        return self.state

    def render(self):
        pass

    def close(self):
        pass

    

class ResSimEnv_v1():

    def __init__(self,
                 grid, k, phi, s_wir, s_oir, # domain properties
                 mu_w, mu_o, mobility, # fluid properties
                 dt, nstep, terminal_step, # timesteps
                 q, s): # initial conditions


        # domain properties
        self.grid=grid
        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        self.k_list = k
        self.phi = phi
        self.s_wir = s_wir
        self.s_oir = s_oir

        # fluid properties
        self.mu_w = mu_w
        self.mu_o = mu_o
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'
        self.mobility = mobility

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.q_init = q # storing inital values for reset function
        self.q = q
        self.s = s

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # Model function (mobility and fractional flow function)
        if mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

        # RL parameters
        self.metadata = {'render.modes': []} # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.reward_range = (0.0, 1.0)       # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.spec = None                     # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py

        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.tol = 1e-5
        self.Q = np.sum(self.q[q>self.tol])                                # total flow across the field
        self.n_inj = self.q[self.q>self.tol].size                          # no of injectors
        self.i_x, self.i_y =  np.where(q>self.tol)[0], np.where(q>self.tol)[1]    # injector co-ordinates
        self.n_prod = self.q[self.q<-self.tol].size                         # no of producers
        self.p_x, self.p_y =  np.where(q<-self.tol)[0], np.where(q<-self.tol)[1]    # producer co-ordinates
        self.action_space = spaces.Box(low=np.array([0.001]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       high=np.array([1]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       dtype=np.float64)

        # for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping_cont(self, action):

        assert all(action>0), 'Invalid action. condition violated: all(action>0) = True'        
        # convert input array into producer/injector 
        inj_flow = action[:self.n_inj] / np.sum(action[:self.n_inj])
        inj_flow = self.Q * inj_flow
        prod_flow = action[self.n_inj:] / np.sum(action[self.n_inj:])
        prod_flow = -self.Q * prod_flow
        
        assert np.sum(inj_flow)>0, 'Invalid action: zero injector flow'
        assert np.sum(prod_flow)<0, 'Invalid action: zero producer flow'

        # add producer/injector flow values
        q = np.zeros(self.grid.shape)
        for i,(x,y) in enumerate( zip(self.i_x, self.i_y) ):
            q[x,y] = inj_flow[i]

        for i,(x,y) in enumerate( zip(self.p_x, self.p_y) ):
            q[x,y] = prod_flow[i]
            
        if np.abs(np.sum(q)) < self.tol:
            q[3,3] = q[3,3] - np.sum(q) # to adjust unbalanced source term in arbitary location in the field due to precision error
        return q

    
    def sim_step(self, q):
        self.q = q
        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        oil_pr = 0.0
        # solve pressure
        self.solverP.s = self.solverS.s
        self.solverP.step()
        self.solverS.v = self.solverP.v
        for _ in range(self.nstep):
            # solve saturation
            self.solverS.step(self.dt)
            self.s_load = self.solverS.s
            oil_pr = oil_pr + -np.sum( self.q[self.q<0] * ( 1- self.f_fn(self.s_load[self.q<0]) ) )*self.dt

        # state
        self.state = self.s_load.reshape(-1)

        #reward
        reward = oil_pr / self.ooip # recovery rate
#         reward = reward*100 # in percentage

        # done
        self.episode_step += 1
        if self.episode_step >= self.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}



    def step(self, action):
        q = self.action_to_q_mapping_cont(action)
        state, reward, done, info = self.sim_step(q)
        return state, reward, done, info

    def set_k(self, k):
        self.k_list = k
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        
    def set_observation_space(self, observation_space):
        self.observation_space = observation_space

    def reset(self):
        self.q = self.q_init
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        self.episode_step = 0
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        return self.state

    def render(self):
        pass

    def close(self):
        pass


class ResSimEnv_v2():

    def __init__(self,
                 grid, k, phi, s_wir, s_oir, # domain properties
                 mu_w, mu_o, mobility, # fluid properties
                 dt, nstep, terminal_step, # timesteps
                 q, s): # initial conditions


        # domain properties
        self.grid=grid
        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        self.k_list = k
        self.phi = phi
        self.s_wir = s_wir
        self.s_oir = s_oir

        # fluid properties
        self.mu_w = mu_w
        self.mu_o = mu_o
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'
        self.mobility = mobility

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.q_init = q # storing inital values for reset function
        self.q = q
        self.s = s

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # Model function (mobility and fractional flow function)
        if mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

        # RL parameters
        self.metadata = {'render.modes': []} # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.reward_range = (0.0, 1.0)       # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.spec = None                     # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py

        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.tol = 1e-5
        self.Q = np.sum(self.q[q>self.tol])                                # total flow across the field
        self.n_inj = self.q[self.q>self.tol].size                          # no of injectors
        self.i_x, self.i_y =  np.where(q>self.tol)[0], np.where(q>self.tol)[1]    # injector co-ordinates
        self.n_prod = self.q[self.q<-self.tol].size                         # no of producers
        self.p_x, self.p_y =  np.where(q<-self.tol)[0], np.where(q<-self.tol)[1]    # producer co-ordinates
        self.action_space = spaces.Box(low=np.array([0.001]*(self.n_prod), dtype=np.float64), 
                                       high=np.array([1]*(self.n_prod), dtype=np.float64), 
                                       dtype=np.float64)

        # for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping_cont(self, action):
        prod_flow = action / np.sum(action)
        prod_flow = -self.Q * prod_flow
        assert np.sum(prod_flow)<0, 'Invalid action: zero producer flow'+str(np.sum(prod_flow))
        q = self.q
        for x,y,i in zip(self.p_x, self.p_y, range(self.n_prod)):
            q[x,y] = prod_flow[i]
        if np.abs(np.sum(q)) < self.tol:
            q[3,3] = q[3,3] - np.sum(q) # to adjust unbalanced source term in arbitary location in the field due to precision error
        return q

    
    def sim_step(self, q):
        self.q = q
        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        oil_pr = 0.0
        for i in range(self.nstep):
            if i%20==0:
                # solve pressure
                self.solverP.s = self.solverS.s
                self.solverP.step()
                self.solverS.v = self.solverP.v
            # solve saturation
            self.solverS.step(self.dt)
            self.s_load = self.solverS.s
            oil_pr = oil_pr + -np.sum( self.q[self.q<0] * ( 1- self.f_fn(self.s_load[self.q<0]) ) )*self.dt

        # state
        self.state = self.s_load.reshape(-1)

        #reward
        reward = oil_pr / self.ooip # recovery rate
#         reward = reward*100 # in percentage

        # done
        self.episode_step += 1
        if self.episode_step >= self.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}



    def step(self, action):
        q = self.action_to_q_mapping_cont(action)
        state, reward, done, info = self.sim_step(q)
        return state, reward, done, info

    def set_k(self, k):
        self.k_list = k
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]

    def reset(self):
        self.q = self.q_init
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        self.episode_step = 0
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        return self.state

    def render(self):
        pass

    def close(self):
        pass
    
    
class ResSimEnv_v3():

    def __init__(self,
                 grid, k, phi, s_wir, s_oir, # domain properties
                 mu_w, mu_o, mobility, # fluid properties
                 dt, nstep, terminal_step, # timesteps
                 q, s): # initial conditions


        # domain properties
        self.grid=grid
        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        self.k_list = k
        self.phi = phi
        self.s_wir = s_wir
        self.s_oir = s_oir

        # fluid properties
        self.mu_w = mu_w
        self.mu_o = mu_o
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'
        self.mobility = mobility

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.q_init = q # storing inital values for reset function
        self.q = q
        self.s = s

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # Model function (mobility and fractional flow function)
        if mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

        # RL parameters
        self.metadata = {'render.modes': []} # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.reward_range = (0.0, 1.0)       # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py
        self.spec = None                     # accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py

        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.tol = 1e-5
        self.Q = np.sum(self.q[q>self.tol])                                # total flow across the field
        self.n_inj = self.q[self.q>self.tol].size                          # no of injectors
        self.i_x, self.i_y =  np.where(q>self.tol)[0], np.where(q>self.tol)[1]    # injector co-ordinates
        self.n_prod = self.q[self.q<-self.tol].size                         # no of producers
        self.p_x, self.p_y =  np.where(q<-self.tol)[0], np.where(q<-self.tol)[1]    # producer co-ordinates
        self.action_space = spaces.Box(low=np.array([0.001]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       high=np.array([1]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       dtype=np.float64)

        # for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping_cont(self, action):

        assert all(action>0), 'Invalid action. condition violated: all(action>0) = True'        
        # convert input array into producer/injector 
        inj_flow = action[:self.n_inj] / np.sum(action[:self.n_inj])
        inj_flow = self.Q * inj_flow
        prod_flow = action[self.n_inj:] / np.sum(action[self.n_inj:])
        prod_flow = -self.Q * prod_flow

        assert np.sum(inj_flow)>0, 'Invalid action: zero injector flow'
        assert np.sum(prod_flow)<0, 'Invalid action: zero producer flow'

        # add producer/injector flow values
        q = np.zeros(self.grid.shape)
        for i,(x,y) in enumerate( zip(self.i_x, self.i_y) ):
            q[x,y] = inj_flow[i]

        for i,(x,y) in enumerate( zip(self.p_x, self.p_y) ):
            q[x,y] = prod_flow[i]
            
        if np.abs(np.sum(q)) < self.tol:
            q[3,3] = q[3,3] - np.sum(q) # to adjust unbalanced source term in arbitary location in the field due to precision error
        return q

    
    def sim_step(self, q):
        self.q = q
        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        oil_pr = 0.0
        for i in range(self.nstep):
            if i%10==0:
                # solve pressure
                self.solverP.s = self.solverS.s
                self.solverP.step()
                self.solverS.v = self.solverP.v
            # solve saturation
            self.solverS.step(self.dt)
            self.s_load = self.solverS.s
            oil_pr = oil_pr + -np.sum( self.q[self.q<0] * ( 1- self.f_fn(self.s_load[self.q<0]) ) )*self.dt

        # state
        self.state = self.s_load.reshape(-1)

        #reward
        reward = oil_pr / self.ooip # recovery rate
#         reward = reward*100 # in percentage

        # done
        self.episode_step += 1
        if self.episode_step >= self.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}



    def step(self, action):
        q = self.action_to_q_mapping_cont(action)
        state, reward, done, info = self.sim_step(q)
        return state, reward, done, info

    def set_k(self, k):
        self.k_list = k
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]

    def reset(self):
        self.q = self.q_init
        self.k = self.k_list[self.np_random.choice(self.k_list.shape[0])]
        self.episode_step = 0
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        return self.state

    def render(self):
        pass

    def close(self):
        pass

