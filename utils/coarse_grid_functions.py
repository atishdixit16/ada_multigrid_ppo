import numpy as np
from numba import jit

@jit(nopython=True)
def hmean(x):
    return (np.sum(x**-1)/x.shape[0])**-1

@jit(nopython=True)
def mean_(x):
    return np.mean(x)

@jit(nopython=True)
def sum_(x):
    return np.sum(x)

@jit(nopython=True)
def get_partition(l,n): 
    p=[0] 
    for i in range(l%n): 
        p.append(p[-1]+l//n+1) 
    for i in range(n-l%n): 
        p.append(p[-1]+l//n) 
    return p 

@jit(nopython=True)
def get_partition_ind(fine_grid_nx, fine_grid_ny, coarse_grid_nx, coarse_grid_ny):
    
    '''
    generate partition indices to be used for upscaling
    
    ''' 
    
    p_1 = get_partition(fine_grid_nx, coarse_grid_nx)
    p_0 = get_partition(fine_grid_ny, coarse_grid_ny) 
    
    return (p_0, p_1)

@jit(nopython=True)
def fine_to_coarse_mapping(fine_array, partition_ind, func):
    
    if func=='mean':
        func_=mean_
    elif func=='hmean':
        func_=hmean
    else:
        func_=sum_
    
    p_0, p_1 = partition_ind[0], partition_ind[1]
    coarse_array = np.empty((len(p_0)-1, len(p_1)-1))
    for i in range(len(p_0)-1):
        for j in range(len(p_1)-1):
            coarse_array[i,j] = func_( np.ascontiguousarray(fine_array[p_0[i]:p_0[i+1], p_1[j]:p_1[j+1]]).reshape(-1) )
    return coarse_array
            
    
@jit(nopython=True)
def coarse_to_fine_mapping(coarse_array, partition_ind):
    p_0, p_1 = partition_ind[0], partition_ind[1]
    fine_array = np.empty((p_0[-1], p_1[-1]))
    for i in range(len(p_0)-1):
        for j in range(len(p_1)-1):
            fine_array[p_0[i]:p_0[i+1], p_1[j]:p_1[j+1]] = coarse_array[i,j]
    return fine_array





# from itertools import product
# import numpy as np


# def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
#     """
#     An accumulation function similar to Matlab's `accumarray` function.
#     source: https://scipy-cookbook.readthedocs.io/items/AccumarrayLike.html

#     Parameters
#     ----------
#     accmap : ndarray
#         This is the "accumulation map".  It maps input (i.e. indices into
#         `a`) to their destination in the output array.  The first `a.ndim`
#         dimensions of `accmap` must be the same as `a.shape`.  That is,
#         `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
#         has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
#         case `accmap[i,j]` gives the index into the output array where
#         element (i,j) of `a` is to be accumulated.  If the output is, say,
#         a 2D, then `accmap` must have shape (15,4,2).  The value in the
#         last dimension give indices into the output array. If the output is
#         1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
#     a : ndarray
#         The input data to be accumulated.
#     func : callable or None
#         The accumulation function.  The function will be passed a list
#         of values from `a` to be accumulated.
#         If None, numpy.sum is assumed.
#     size : ndarray or None
#         The size of the output array.  If None, the size will be determined
#         from `accmap`.
#     fill_value : scalar
#         The default value for elements of the output array. 
#     dtype : numpy data type, or None
#         The data type of the output array.  If None, the data type of
#         `a` is used.

#     Returns
#     -------
#     out : ndarray
#         The accumulated results.

#         The shape of `out` is `size` if `size` is given.  Otherwise the
#         shape is determined by the (lexicographically) largest indices of
#         the output found in `accmap`.


#     Examples
#     --------
#     >>> from numpy import array, prod
#     >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
#     >>> a
#     array([[ 1,  2,  3],
#            [ 4, -1,  6],
#            [-1,  8,  9]])
#     >>> # Sum the diagonals.
#     >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
#     >>> s = accum(accmap, a)
#     array([9, 7, 15])
#     >>> # A 2D output, from sub-arrays with shapes and positions like this:
#     >>> # [ (2,2) (2,1)]
#     >>> # [ (1,2) (1,1)]
#     >>> accmap = array([
#             [[0,0],[0,0],[0,1]],
#             [[0,0],[0,0],[0,1]],
#             [[1,0],[1,0],[1,1]],
#         ])
#     >>> # Accumulate using a product.
#     >>> accum(accmap, a, func=prod, dtype=float)
#     array([[ -8.,  18.],
#            [ -8.,   9.]])
#     >>> # Same accmap, but create an array of lists of values.
#     >>> accum(accmap, a, func=lambda x: x, dtype='O')
#     array([[[1, 2, 4, -1], [3, 6]],
#            [[-1, 8], [9]]], dtype=object)
#     """

#     # Check for bad arguments and handle the defaults.    
#     if accmap.shape[:a.ndim] != a.shape:
#         raise ValueError("The initial dimensions of accmap must be the same as a.shape")
#     if func is None:
#         func = np.sum
#     if dtype is None:
#         dtype = a.dtype
#     if accmap.shape == a.shape:
#         accmap = np.expand_dims(accmap, -1)
#     adims = tuple(range(a.ndim))
#     if size is None:
#         size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
#     size = np.atleast_1d(size)

#     # Create an array of python lists of values.
#     vals = np.empty(size, dtype='O')
#     for s in product(*[range(k) for k in size]):
#         vals[s] = []
#     for s in product(*[range(k) for k in a.shape]):
#         indx = tuple(accmap[s])
#         val = a[s]
#         vals[indx].append(val)

#     # Create the output array.
#     out = np.empty(size, dtype=dtype)
#     for s in product(*[range(k) for k in size]):
#         if vals[s] == []:
#             out[s] = fill_value
#         else:
#             out[s] = func(vals[s])

#     return out

# def get_accmap(grid,nx_,ny_):
    
#     '''
#     generate accmap to be used for upscaling
    
#     grid: original grid
#     nx_: new coarse nx
#     ny_: new coarse ny
    
#     '''
    
#     ind_x = np.sort(np.array(list(range(grid.ny)))%ny_)
#     ind_y = np.sort(np.array(list(range(grid.nx)))%nx_)
#     xv,yv = np.meshgrid(ind_x,ind_y)
#     accmap = np.dstack((xv.reshape(grid.nx,grid.ny,1),yv.reshape(grid.nx,grid.ny,1))) 
    
#     return accmap.transpose((1,0,2))

# def fine_to_coarse_mapping(fine_array, accmap, func):
#     return accum(accmap,fine_array, func)

# def coarse_to_fine_mapping_old(coarse_array,accmap):
#     fine_nx, fine_ny = accmap.shape[0], accmap.shape[1]
#     fine_array = np.zeros((fine_ny, fine_nx))
#     for i in range(fine_ny):
#         for j in range(fine_nx):
#             ind_x, ind_y = accmap[i,j][0], accmap[i,j][1]
#             fine_array[i,j] = coarse_array[ind_x,ind_y]
#     return fine_array

# def coarse_to_fine_mapping(coarse_array,accmap): #  old copy used in square domains
#     fine_nx, fine_ny = accmap.shape[0], accmap.shape[1]
#     fine_array = np.zeros((fine_nx, fine_ny))
#     for i in range(fine_nx):
#         for j in range(fine_ny):
#             ind_x, ind_y = accmap[i,j][0], accmap[i,j][1]
#             fine_array[i,j] = coarse_array[ind_x,ind_y]
#     return fine_array