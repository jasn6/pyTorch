# PyTorch has a built-in differentiation engine called torch.autograd. 
# It supports automatic computation of gradient for any computational graph.

import torch

# In this network, w and b are parameters, which we need to optimize. 
# Thus, we need to be able to compute the gradients of loss function with respect to those variables. 
# In order to do that, we set the requires_grad property of those tensors.
# You can set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True) method.
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# computing the function in the forward direction, and its derivative during the backward propagation step
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients
loss.backward()
print(w.grad)
print(b.grad)

# We can only obtain the grad properties for the leaf nodes of the computational graph, 
# which have requires_grad property set to True. For all other nodes in our graph, gradients will not be available.

# We can only perform gradient calculations using backward once on a given graph, for performance reasons. 
# If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.