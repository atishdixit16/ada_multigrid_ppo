import numpy as np

def eval_actions(env, actions_, base_steps=0):
    obs, done = env.reset(), False
    rewards, actions, states = [],[],[]
    i=0
    while not done:
        action = actions_[i]
        if i <= (base_steps-1):
            action = np.array([1]*env.action_space.shape[0])
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(env.s_load.copy())
        actions.append(env.q.copy())
        i=i+1
    return states, actions, rewards 

def eval_model(env, model, base_steps=0):
    obs, done = env.reset(), False
    rewards, actions, states = [],[],[]
    i=0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        if i <= (base_steps-1):
            action = np.array([1]*env.action_space.shape[0])
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(env.s_load.copy())
        actions.append(env.q.copy())
        i=i+1
    return states, actions, rewards

def eval_policy(env, policy, base_steps=0):
    obs, done = env.reset(), False
    rewards, actions, states = [],[],[]
    i=0
    while not done:
        action = policy(env)
        if i <= (base_steps-1):
            action = np.array([1]*env.action_space.shape[0])
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(env.s_load.copy())
        actions.append(env.q.copy())
        i=i+1
    return states, actions, rewards

def eval_generic_policy(env, policy, base_steps=0):
    if callable(policy):
        return eval_policy(env, policy, base_steps)
    return eval_model(env, policy, base_steps)
        

def eval_frac_flow(env, actions_, x_loc, y_loc):
    obs, done = env.reset(), False
    ff_array = []
    i=0
    while not done:
        action = actions_[i]
        obs, reward, done, info = env.step(action)
        ff_values = []
        for x,y in zip(x_loc, y_loc):
            ff_values.append(env.f_fn(env.s_load[x,y]))
        i=i+1
        ff_array.append(ff_values)
    return np.array(ff_array)