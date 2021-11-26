import numpy as np
from tqdm.notebook import trange, tqdm
from utils.env_evaluate_functions import eval_actions, eval_frac_flow



def get_ff_coords(k_samples, env, x_loc, y_loc):

    actions_base = np.array([ np.append(env.q[env.q>1e-5], -env.q[env.q<-1e-5]) for _ in range(env.terminal_step) ])
    actions_base = actions_base.reshape(env.terminal_step, -1)

    ff_coords = []
    for k in tqdm(k_samples):
        env.set_k(np.array([k]))
        ff_array = eval_frac_flow(env, actions_base, x_loc=x_loc, y_loc=y_loc)
        ff_coords.append(ff_array)
    ff_coords = np.array(ff_coords)
    return ff_coords


def connectivity_distance(ff_i, ff_j): # http://pangea.stanford.edu/~jcaers/Thesis_PhD_KPark.pdf (eq. 2.43)
    distance=0
    for ff_pi, ff_pj in zip(ff_i, ff_j):
        distance = distance + np.sum((ff_pi - ff_pj)**2)
    return distance


def connectivity_dist_mat(k_samples, ff_coords):
    dist_matrix_connectivity = np.zeros((k_samples.shape[0], k_samples.shape[0]))
    for i,ff_i in tqdm(enumerate(ff_coords)):
        for j,ff_j in enumerate(ff_coords):
            if j>=i:
                dist_matrix_connectivity[i,j] = connectivity_distance(ff_i, ff_j)
                dist_matrix_connectivity[j,i] = dist_matrix_connectivity[i,j]
    return dist_matrix_connectivity


def get_connectivity_dist_mat(k_samples, env, x_loc, y_loc):
    print('compute connectivity distances...')
    ff_coords = get_ff_coords(k_samples, env, x_loc, y_loc)
    print('form distance matrix...')
    cdm = connectivity_dist_mat(k_samples, ff_coords)
    return cdm


def get_euclidean_dist_mat(k_samples):
    dist_matrix_euclidian = np.zeros((k_samples.shape[0], k_samples.shape[0]))
    for i,k_i in tqdm(enumerate(k_samples)):
        for j,k_j in enumerate(k_samples):
            if j>=i:
                dist_matrix_euclidian[i,j] = np.linalg.norm(k_i-k_j) 
                dist_matrix_euclidian[j,i] = dist_matrix_euclidian[i,j]
    return dist_matrix_euclidian

