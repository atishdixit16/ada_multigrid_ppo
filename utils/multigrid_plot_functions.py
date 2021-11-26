import numpy as np
from utils.data_handling import get_episode_reward_data
from utils.plot_functions import plot_learning_tr

def get_eq_fine_data(t_array, r_array, findelity_jump_indices, equivalent_fine_runtime):
    delta = t_array[1] - t_array[0]
    deltas = np.array([])
    for i,p in enumerate(equivalent_fine_runtime):
        delta_ = np.array([p]*(findelity_jump_indices[i+1] - findelity_jump_indices[i]))
        deltas = np.append(deltas, delta_)
    t_fine = np.cumsum(deltas*delta )
    
    ts, rs = [], []
    for i,p in enumerate(equivalent_fine_runtime):
        ts.append(t_fine[findelity_jump_indices[i]:findelity_jump_indices[i+1]+1])
        rs.append(r_array[findelity_jump_indices[i]:findelity_jump_indices[i+1]+1])
    
    return ts, rs

def singlegrid_learning_plots( axis,
                               log_dir_array,
                               grid_fidelity_array,
                               episode_limit_array,
                               seed_no,
                               result_type,
                               x_label,
                               y_label,
                               title):
    
    for sg_dir in log_dir_array:
        # get learning plot data
        t_ppo_1_train, r_ppo_1_train = get_episode_reward_data(sg_dir, 
                                                               seeds=[seed_no], 
                                                               result_type=result_type, 
                                                               func_type='mean')
        # plot learning data
        plot_learning_tr([np.cumsum(t_ppo_1_train)], 
                         [r_ppo_1_train], 
                         axis, legends=[''], window=1,
                         x_label=x_label,
                         y_label=y_label)
        
    legends = []    
    for beta in grid_fidelity_array:
        legends.append(r'$\beta='+str(beta)+'$')
        
    axis.legend(legends)
    axis.set_xticks(np.cumsum(np.insert(episode_limit_array,0,0)))
    axis.set_title(title)
    axis.margins(0.03)
    
    return 
    
    
def multigrid_learning_plots( axis,
                              log_dir,
                              grid_fidelity_array,
                              episode_limit_array,
                              equivalent_fine_runtime_array,
                              fidelity_indices,
                              seed_no,
                              result_type,
                              x_label,
                              y_label,
                              x_label_2,
                              title):
    
    # get data for learning plot 
    t_ppo_mg_train, r_ppo_mg_train = get_episode_reward_data(log_dir, 
                                                             seeds=[seed_no], 
                                                             result_type=result_type, 
                                                             func_type='mean')
    t_ppo_mg_train = np.cumsum(t_ppo_mg_train)
    fidelity_indices = np.append(fidelity_indices, t_ppo_mg_train.shape[0]) 
    ts, rs = get_eq_fine_data(t_ppo_mg_train, r_ppo_mg_train, fidelity_indices, [1,1,1])
        
    # plot learning data
    for t,r in zip(ts, rs):
        plot_learning_tr([t], 
                         [r], 
                         axis, legends=[''], window=1,
                         x_label=x_label,
                         y_label=y_label)
        
    legends = []    
    for beta in grid_fidelity_array:
        legends.append(r'$\beta='+str(beta)+'$')
    axis.legend(legends)
    axis.set_title(title)
    axis.margins(0.03)
    tick_location = [0]
    for t in ts:
        tick_location.append(t[-1])
    axis.set_xticks(tick_location)
    
    # add scondary x-axis for equivalent fine episodes
    ts_, rs_ = get_eq_fine_data(t_ppo_mg_train, r_ppo_mg_train, fidelity_indices, equivalent_fine_runtime_array)
    ax_x2 = axis.twiny()
    ax_x2.xaxis.set_ticks_position("bottom")
    ax_x2.xaxis.set_label_position("bottom")
    ax_x2.spines["bottom"].set_position(("axes", -0.2))
    ax_x2.set_xlabel(x_label_2)
    ax_x2.set_xticks(tick_location)
    tick_labels = ['0']
    for t in ts_:
        tick_labels.append(str(int(t[-1])))
    ax_x2.set_xticklabels(tick_labels)
    ax_x2.margins(0.03)
    
    return
    