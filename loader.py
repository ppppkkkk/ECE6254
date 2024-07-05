import numpy as np

# Load the numpy array from the file
feats_npy = np.load(r'B:\GAlign\graph_data\allmv_tmdb\tmdb\graphsage\feats.npy')
print(feats_npy)
# Let's take a look at the shape and data type of the numpy array
array_shape = feats_npy.shape
array_dtype = feats_npy.dtype

print(array_shape, array_dtype)
