import numpy as np

'''
generates permeability field with a linear channel across the grid
start and end of the channel are selected with uniform distribution

parameters:
nx, ny, lx, ly: grid dimensions
channel_k: permeability value at the channel
base_k: permeability value outside the channel
channel_width: width of the permeability channel (fraction of ly)
sample_size: number of realizations generated
seed: seed for reproducibility

'''

def get_channel_end_indices(nx=32, ny=32, lx=1.0, ly=1.0, channel_width=0.125, seed=1):
    assert channel_width<1.0 and channel_width>0.0, 'invalid channel width. condition violated: 0 < channel_width < 1'
    channel_left_end = np.random.uniform(0,(1.0-channel_width))
    channel_right_end = np.random.uniform(0,(1.0-channel_width))
    return channel_left_end, channel_right_end


def single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=0.4375, channel_right_end=0.4375):
    index_left = round(channel_left_end*ny)
    index_right = round(channel_right_end*ny)
    grid_channel_width = round(channel_width*ny)
    k = base_k*np.ones((nx,ny))
    for i in range(nx):
        j = ( (index_right - index_left) / nx ) *i + index_left
        for w in range(grid_channel_width ):
            k[round(j)+w, i] = channel_k
    return k


def batch_generate(nx=32, ny=32, lx=1.0, ly=1.0, channel_k=1.0, base_k=0.01, channel_width_range=(0.1,0.3), sample_size=10, seed=1):
    np.random.seed(seed) #for reproducibility
    k_batch = []
    for _ in range(sample_size):
        channel_width = np.random.uniform(channel_width_range[0], channel_width_range[1]) 
        channel_left_end, channel_right_end = get_channel_end_indices(nx, ny, lx, ly, channel_width, seed)
        k = single_generate(nx,ny,lx,ly,channel_k, base_k, channel_width, channel_left_end, channel_right_end)
        k_batch.append(k)
    return np.array(k_batch)

def generate_domain_based_train_data(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width_range=(0.1,0.3), seed=1 ):
    np.random.seed(seed) #for reproducibility
    end_values = [0.125, 0.4375, 0.75 ]
    k_batch = []
    for l_end in end_values:
        for r_end in end_values:
            c_width = np.random.uniform(channel_width_range[0], channel_width_range[1]) 
            k = single_generate(nx=nx,ny=ny,lx=lx,ly=ly,
                                channel_k=channel_k, base_k=base_k, 
                                channel_width=c_width, channel_left_end=l_end, channel_right_end=r_end)
            k_batch.append(k)
    return np.array(k_batch)


