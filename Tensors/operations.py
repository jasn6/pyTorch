import torch

# Attributes of a Tensor

shape = (3,2)
tensor = torch.rand(shape)

# Shape is a tuple of tensor dimensions 
print(f"Shape of a Tensor: {tensor.shape}")

# Datatype is the type of numbers in the Tensor
print(f"Datatype of a Tensor: {tensor.dtype}")

# Device tensor is stored on 
print(f"Tensor is stored on: {tensor.device}")

# Operations on Tensors

# Tensor indexing and slicing
tensor2 = torch.rand(4,4)
print(f"First row: {tensor2[0]}")
print(f"First column: {tensor2[:,0]}")
print(f"Last column: {tensor2[..., -1]}")
tensor2[:,1] = 0
print(tensor2)

# Joining Tensors
t1 = torch.cat([tensor2, tensor2, tensor2], dim=1)
print(t1)

# Arithmetic operations

# Matrix multiplying:
# tensor.T is the transpose of the tensor
t1 = tensor2 @ tensor2.T
print(t1)
t2 = tensor2.matmul(tensor2.T)
print(t2)
# create a tensor of size t1 and output it to t3
t3 = torch.rand_like(t1)
torch.matmul(tensor2, tensor2.T, out=t3)

# Element-wise product
z1 = tensor2 * tensor2
print(z1)
z2 = tensor2.mul(tensor2)
print(z2)
z3 = torch.rand_like(tensor2)
torch.mul(tensor2, tensor2, out=z3)

# Convert single-element tensors to python numerical values using item()
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations that store the result into the operand, they are denoted by a _ suffix
# add_ or copy_ or t_ will all change the object
tensor2.add_(5)