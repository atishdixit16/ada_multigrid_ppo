import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import re
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_s_animation(states, actions, rewards, 
                     s_min=0.0, s_max=1.0, levels=10, 
                     marker_size_ref=3, 
                     pause_time=0.5, figsize_scale=6, show_wells=True):
    aspect = states[0].shape[1] / states[0].shape[0]
    fig, axs = plt.subplots(1,1,figsize=(round(figsize_scale*aspect),figsize_scale) )
    axs.axis('off')
    reward = round(sum(rewards*100))
    for i,(state, action) in enumerate(zip(states, actions)):
        im = axs.contourf(state, vmin=s_min, vmax=s_max, levels=levels, cmap='RdBu')
        if show_wells:
            # producers
            ys, xs = np.where(action < -1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(-action[y,x]*marker_size_ref)
                axs.scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
            # injectors
            ys, xs = np.where(action > 1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(action[y,x]*marker_size_ref)
                axs.scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
        fig.canvas.draw()
        axs.set_title(f'step {i+1}: RF: {reward} %')
        sleep(pause_time)
    return fig

def plot_s_snapshots(states, actions, rewards, 
                     s_min=0.0, s_max=1.0, levels=10, 
                     marker_size_ref=3, figsize_scale=3, show_wells=True):
    aspect = states[0].shape[1] / states[0].shape[0]
    ctrl_steps = len(states)
    fig, axs = plt.subplots(1,ctrl_steps,figsize=(round(figsize_scale*aspect)*ctrl_steps,figsize_scale) )
    for i,(ax, state, action) in enumerate(zip(axs, states, actions)):
        ax.axis('off')
        reward = round(sum(rewards[:(i+1)]*100))
        im = ax.contourf(state, vmin=s_min, vmax=s_max, levels=levels, cmap='RdBu')
        if show_wells:
            # producers
            ys, xs = np.where(action < -1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(-action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
            # injectors
            ys, xs = np.where(action > 1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
        ax.set_title(f'step {i+1}: RF: {reward} %')
#         fig.canvas.draw()
    return fig

def plot_s_snapshots_with_k(k, states, actions, rewards, 
                            s_min=0.0, s_max=1.0, levels=10, 
                            marker_size_ref=3, figsize_scale=3, show_wells=True):
    aspect = states[0].shape[1] / states[0].shape[0]
    ctrl_steps = len(states)
    fig, axs = plt.subplots(1,ctrl_steps+1,figsize=(round(figsize_scale*aspect)*ctrl_steps,figsize_scale) )
    
    
    axs[0].axis('off')
    im = axs[0].contourf(k, cmap='viridis')
    axs[0].set_title(f'log(k)\n')
    if show_wells:
        # producers
        ys, xs = np.where(actions[0] < -1e-5)
        for x,y in zip(xs, ys):
            marker_size = int(-actions[0][y,x]*marker_size_ref)
            axs[0].scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
        # injectors
        ys, xs = np.where(actions[0] > 1e-5)
        for x,y in zip(xs, ys):
            marker_size = int(actions[0][y,x]*marker_size_ref)
            axs[0].scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
    
    
    for i,(ax, state, action) in enumerate(zip(axs[1:], states, actions)):
        ax.axis('off')
        reward = round(sum(rewards[:(i+1)]*100))
        im = ax.contourf(state, vmin=s_min, vmax=s_max, levels=levels, cmap='RdBu')
        if show_wells:
            # producers
            ys, xs = np.where(action < -1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(-action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
            # injectors
            ys, xs = np.where(action > 1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
        ax.set_title(f'step {i+1} \n RF: {reward} %')
#         fig.canvas.draw()
    return fig

def plot_s(states, actions, rewards, ax,
           s_min=0.0, s_max=1.0, levels=10, 
           marker_size_ref=3, figsize_scale=3, show_wells=True,
           end_step=5):
    aspect = states[0].shape[1] / states[0].shape[0]
    ctrl_steps = len(states)
    for i,(state, action) in enumerate(zip(states, actions)):
        ax.axis('off')
        reward = round(sum(rewards[:(i+1)]*100))
        im = ax.contourf(state, vmin=s_min, vmax=s_max, levels=levels, cmap='RdBu')
        if show_wells:
            # producers
            ys, xs = np.where(action < -1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(-action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
            # injectors
            ys, xs = np.where(action > 1e-5)
            for x,y in zip(xs, ys):
                marker_size = int(action[y,x]*marker_size_ref)
                ax.scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
        ax.set_title(f'step {i+1}: RF: {reward} %')
#         fig.canvas.draw()
        if i==(end_step-1):
            break
    return ax


def plot_k_array(k_train, q, rows=3, cols=3, marker_size=5, value_range=None, cbar_axs=None, fig_size=(6,6)):
    if q is not None:
        # injectors
        coords_i = np.argwhere(q>1e-5)
        # producers
        coords_p = np.argwhere(q<-1e-5)
    if value_range is None:
        max_value = np.max(k_train)
        min_value = np.min(k_train)
    else:
        max_value=value_range[1]
        min_value=value_range[0]
    fig, axs = plt.subplots(rows,cols,figsize=fig_size)
    plt.subplots_adjust(top=None, hspace=0.5)
    for i, (ax, k) in enumerate(zip(axs.ravel(), k_train)):
        # ax.contourf(k, levels=10)
        im = ax.imshow(k, vmin=min_value, vmax=max_value, origin='lower', cmap='viridis')
        # print(k.max(), k.min())
        if q is not None:
            ax.scatter(coords_i[:, 1] ,coords_i[:, 0], marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
            ax.scatter(coords_p[:, 1] ,coords_p[:, 0], marker='o', c='indianred', edgecolor='black', s=marker_size)
        ax.set_title('index: {}'.format(i))
        ax.axis('off')
    if cbar_axs is None:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    else:
        cbar_ax = fig.add_axes(cbar_axs)
    fig.colorbar(im, cax=cbar_ax)    
    return fig

def plot_k(k, q, axs, title='k-plot', marker_size=10, value_range=None):
    if value_range is None:
        max_value = np.max(k)
        min_value = np.min(k)
    else:
        max_value=value_range[1]
        min_value=value_range[0]
#     print(max_value, min_value)
    im = axs.imshow(k, vmin=min_value, vmax=max_value, origin='lower', cmap='viridis')
#     print(k.max(), k.min())
    if q is not None:
        # injectors
        coords_i = np.argwhere(q>1e-5)
        # producers
        coords_p = np.argwhere(q<-1e-5)
        axs.scatter(coords_i[:, 1] ,coords_i[:, 0], marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
        axs.scatter(coords_p[:, 1] ,coords_p[:, 0], marker='o', c='indianred', edgecolor='black', s=marker_size)
    axs.set_title(title)
    axs.axis('off')    
    return im

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve', window=50):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def plot_learning(log_dir, case='train', multigrid='False'):
    with np.load(log_dir+'/results_'+case+'/evaluations.npz') as data:
        t = data['timesteps']
        r = data['results']
    fig, axs = plt.subplots(1,1,figsize=(6,5))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=None, hspace=None)
    if multigrid:
        axs.plot(r)
    else:
        axs.plot(t,r)
    axs.grid('on', alpha=0.3)
    axs.set_xlabel('number of timesteps')
    axs.set_ylabel('rewards')
    return fig

def plot_learning_tr(ts,rs, axs,legends=None, ref_value=None, window=5, 
                     x_label='number of timesteps',
                     y_label='rewards'):
    for t,r in zip(ts, rs):
        r = moving_average(r, window=window)
        t = t[len(t) - len(r):]
        axs.plot(t,r)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.grid('on', alpha=0.3)
    if ref_value is not None:
        axs.hlines(ref_value, xmin=np.min(t), xmax=np.max(t), color='gray', linestyles='dashed')
        if legends is not None:
            legends.append('DE')
    if legends is not None:
        axs.legend(legends)
    return 

def plot_rl(log_dirs, legends, ref_value=None):
    ts, rs = [], []
    for log_dir in log_dirs:
        with np.load(log_dir+'/evaluations.npz') as data:
            ts.append(data['timesteps'])
            rs.append(data['results'])
    fig, axs = plt.subplots(1,1,figsize=(6,5))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=None, hspace=None)
    for t,r in zip(ts, rs):
        r = moving_average(r, window=5)
        t = t[len(t) - len(r):]
        axs.plot(t,r)
    axs.set_xlabel('number of timesteps')
    axs.set_ylabel('rewards')
    axs.grid('on', alpha=0.3)
    if ref_value is not None:
        axs.hlines(ref_value, xmin=np.min(t), xmax=np.max(t), color='gray', linestyles='dashed')
        legends.append('DE')
    axs.legend(legends)
    return fig

def plot_actions(r_array, a_array, s_array, case_names, 
                 s_min=0.2, s_max=0.8, 
                 levels=10, show_wells=True, marker_size_ref=0.3,
                 time_per_step=np.nan, time_unit='day',
                 figsize=None, cbar_axs=None):
    cases = len(r_array)
    steps = len(r_array[0])
    if figsize is None:
        fig, axs = plt.subplots(cases,steps, figsize=(9,2*cases), sharex=True)
    else:
        fig, axs = plt.subplots(cases,steps, figsize=figsize, sharex=True)
    
    for i,(state_array, action_array, reward_array, title) in enumerate( zip(s_array, a_array, r_array, case_names) ):
        for j,( state, action) in enumerate(zip(state_array, action_array)):
            axs[i,j].axis('off')
            reward = round(sum(reward_array[:(j+1)]*100))
            im = axs[i,j].contourf(state, vmin=s_min, vmax=s_max, levels=levels, cmap='RdBu')
            if show_wells:
                # producers
                ys, xs = np.where(action < -1e-5)
                for x,y in zip(xs, ys):
                    marker_size = int(-action[y,x]*marker_size_ref)
                    axs[i,j].scatter(x ,y, marker='o', c='indianred', edgecolor='black', s=marker_size)
                # injectors
                ys, xs = np.where(action > 1e-5)
                for x,y in zip(xs, ys):
                    marker_size = int(action[y,x]*marker_size_ref)
                    axs[i,j].scatter(x ,y, marker='o',  c='cornflowerblue', edgecolor='black', s=marker_size)
            if j==0:
                title_ = re.sub(r'([^a-zA-Z0-9\s]+?)', ' ', title)
                axs[i,j].text(-35,state.shape[1]/2,title_)
            axs[i,j].set_title(f'{time_unit} {(j+1)*time_per_step}: {int(reward)} \% RF', fontsize=10)
    if cbar_axs is None:
        cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    else:
        cbar_ax = fig.add_axes(cbar_axs)
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal") 
    return fig
            

def plot_rewards(r_array_eval, axs, model_names):
    rs=[]
    total_plots = 2+len(model_names)
    for i in range(total_plots):
        r=[]
        for r_array in r_array_eval:
            r.append(sum(r_array[i]*100))
        rs.append(r)

    indices = np.arange(len(rs[0]))
    sort_indices = np.argsort(rs[0] )
    # print(sort_indices)
    for r in rs[2:]:
        axs.plot(np.array(r)[sort_indices], 'o--')
    axs.plot(np.array(rs[1])[sort_indices], '.--', color='gray')
    axs.plot(np.array(rs[0])[sort_indices], '.--', color='brown')
    axs.legend(model_names+['DE', 'Base'])
    axs.set_xlabel('evaluation sample index')
    axs.set_ylabel('recovery factor \%')
    axs.set_xticks(indices)
    axs.set_xticklabels(sort_indices)
    axs.set_title('Optimisation Results for Evaluation Permeabilities')
    axs.grid('on', alpha=0.3)
    return 

def plot_rewards_wo_de(r_array_eval, axs, model_names):
    rs=[]
    total_plots = 1+len(model_names)
    for i in range(total_plots):
        r=[]
        for r_array in r_array_eval:
            r.append(sum(r_array[i]*100))
        rs.append(r)

    indices = np.arange(len(rs[0]))
    sort_indices = np.argsort(rs[0] )
    # print(sort_indices)
    for r in rs[1:]:
        axs.plot(np.array(r)[sort_indices], 'o--')
    axs.plot(np.array(rs[0])[sort_indices], '.--', color='brown')
    axs.legend(model_names+['Base'])
    axs.set_xlabel('evaluation sample index')
    axs.set_ylabel('recovery factor \%')
    axs.set_xticks(indices)
    axs.set_xticklabels(sort_indices)
    axs.set_title('Optimisation Results for Evaluation Permeabilities')
    axs.grid('on', alpha=0.3)
    return 