# to access functions from other locations
import sys
sys.path.append('/data/ad181/RemoteDir/k_variability_in_ressim_env/SPE10_like_envs/')

import numpy as np
from utils.env_evaluate_functions import eval_actions, eval_model, eval_generic_policy

def get_mean_tr_data(case_dir, seeds, result_type='train', episode_len=1):
    log_dirs=[]
    for seed in seeds:
        log_dir = case_dir+'/seed_'+str(seed)+'/results_'+result_type
        log_dirs.append(log_dir)
        
    ts, rs = [],[]
    for log_dir in log_dirs:
        with np.load(log_dir+'/evaluations.npz') as data:
            ts.append(data['timesteps'])
            rs.append(data['results'])
    rs = np.array(rs).reshape(len(seeds),-1)
    t_data = np.array(ts[0])/episode_len
    r_data = np.mean(rs, axis=0)
    return t_data, r_data

def get_episode_reward_data(log_dir, seeds, result_type, func_type='mean'):
    
    log_dirs=[]
    for seed in seeds:
        dir_ = log_dir+'/seed_'+str(seed)+'/results_'+result_type
        log_dirs.append(dir_)
        
    ts, rs = [],[]
    for log_dir in log_dirs:
        with np.load(log_dir+'/evaluations.npz') as data:
            ts.append(data['timesteps'])
            rs.append(data['results'])
            episode_len = data['ep_lengths'][0]
    rs = np.array(rs).reshape(len(seeds),-1)
    
    episodes = np.array(ts[0])/episode_len
    
    if func_type=='mean':
        rewards =  np.mean(rs, axis=0)
        
    if func_type=='median':
        rewards =  np.median(rs, axis=0)
    
    return episodes, rewards

def get_sar_data(envs, rl_k_eval_indices, opt_dir, models, model_names, model_base_steps = 0):
    
    print('warning: Data gathering in progress. might take few minutes..')
    
    case_names_eval, r_array_eval, a_array_eval, s_array_eval = [],[],[],[]
    base_actions = np.ones((envs[0].terminal_step,envs[0].action_space.shape[0]))
    terminal_steps = envs[0].terminal_step

    for i,idx in enumerate(rl_k_eval_indices):
    
        opt_actions = np.load(opt_dir+'/ck_argmax_'+str(idx)+'.npy')
        opt_actions = opt_actions.reshape(terminal_steps, -1)

        states_base, actions_base, r_base = eval_actions(envs[idx], base_actions)
        states_opt, actions_opt, r_opt = eval_actions(envs[idx], opt_actions, base_steps=model_base_steps)
        s_array = [states_base, states_opt]
        r_array = [r_base, r_opt]
        a_array = [actions_base, actions_opt]
        case_names = ['Base_'+str(i), 'DE_'+str(i) ]
        for model,name in zip(models, model_names):
            states, actions, rewards = eval_generic_policy(envs[idx], model, base_steps=model_base_steps)
            s_array.append(states)
            r_array.append(rewards)
            a_array.append(actions)
            case_names.append(name+'_'+str(i))
        
        case_names_eval.append(case_names)
        r_array_eval.append(r_array)
        a_array_eval.append(a_array)
        s_array_eval.append(s_array)
        
    return s_array_eval, a_array_eval, r_array_eval, case_names_eval


def get_sar_data_wo_de(envs, rl_k_eval_indices, models, model_names, model_base_steps = 0):
    
    print('warning: Data gathering in progress. might take few minutes..')
    
    case_names_eval, r_array_eval, a_array_eval, s_array_eval = [],[],[],[]
    base_actions = np.ones((envs[0].terminal_step,envs[0].action_space.shape[0]))
    terminal_steps = envs[0].terminal_step

    for i,idx in enumerate(rl_k_eval_indices):

        states_base, actions_base, r_base = eval_actions(envs[idx], base_actions)
        s_array = [states_base]
        r_array = [r_base]
        a_array = [actions_base]
        case_names = ['Base_'+str(i) ]
        for model,name in zip(models, model_names):
            states, actions, rewards = eval_generic_policy(envs[idx], model, base_steps=model_base_steps)
            s_array.append(states)
            r_array.append(rewards)
            a_array.append(actions)
            case_names.append(name+'_'+str(i))
        
        case_names_eval.append(case_names)
        r_array_eval.append(r_array)
        a_array_eval.append(a_array)
        s_array_eval.append(s_array)
        
    return s_array_eval, a_array_eval, r_array_eval, case_names_eval

def get_sar_data_wo_base_de(envs, rl_k_eval_indices, models, model_names, model_base_steps = 0):
    
    print('warning: Data gathering in progress. might take few minutes..')
    
    case_names_eval, r_array_eval, a_array_eval, s_array_eval = [],[],[],[]
    base_actions = np.ones((envs[0].terminal_step,envs[0].action_space.shape[0]))
    terminal_steps = envs[0].terminal_step

    for i,idx in enumerate(rl_k_eval_indices):
        s_array = []
        r_array = []
        a_array = []
        case_names = []
        for model,name in zip(models, model_names):
            states, actions, rewards = eval_generic_policy(envs[idx], model, base_steps=model_base_steps)
            s_array.append(states)
            r_array.append(rewards)
            a_array.append(actions)
            case_names.append(name+'_'+str(i))
        
        case_names_eval.append(case_names)
        r_array_eval.append(r_array)
        a_array_eval.append(a_array)
        s_array_eval.append(s_array)
        
    return s_array_eval, a_array_eval, r_array_eval, case_names_eval
