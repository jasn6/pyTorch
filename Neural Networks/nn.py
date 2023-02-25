# In pyTorch a neural network is a module itself that consists of other modules (layers). 
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a neural network by subclassing nn.module
# Initialize the neural network layers in __init__
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    # operations on intput data
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = "cpu"
model = NeuralNetwork().to(device)
print(model)

# To use the model, need to pass in input data, which will then execute the model's forward function
# Do not call model.forward() directly

# Model Layers 

# 3 images of size 28 x 28 
input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.Flatten() converts a 2D image into a contiguous array of pixel values
# i.e 28x28 2D image becomes a contiguous array of 784 pixel values
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Linear Layer, nn.Linear() applies a linear transformation on the input using its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU 
# Non-linear activations are what create the complex mappings between the model’s inputs and outputs. 
# They are applied after linear transformations to introduce nonlinearity. 
# Fits the large number of all weights activations into a certain range
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential is an ordered container of modules.
# The data is passed through all the modules in the same order as defined. 
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module. 
# The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class.
# dim parameter indicates the dimension along which the values must sum to 1.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)

# Model Parameters, layers inside a neural network are parameterized.
# Subclassing nn.Module automatically tracks all fields defined inside your model object, 
# and makes all parameters accessible using your model’s parameters() or named_parameters() methods
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")