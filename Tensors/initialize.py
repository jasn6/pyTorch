import torch
import numpy as np

# Different ways to Initialize a Tensor

# Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

# From a numpy array
np_array = np.array(data)
x_data = torch.from_numpy(np_array)

# From another tensor, will retain the shape and datatype of the argument tensor
x_ones = torch.ones_like(x_data)
print(x_ones)
# Overwritting the argument tensors datatype 
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

# With random or constant values
shape = (3,4,)
random_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
print(random_tensor)
print(ones_tensor)
