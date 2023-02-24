import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


# Transforms allow for manipulation of data, mainly used to make data more suitable for training.

# TorchVision datasets have two transform parameters -transform to modfiy the features and
# -target_transform to modify the labels, it accepts callables containing the transformation logic

# For training we need the features as normalized tensors, and the labels as one-hot encoded tensors. 


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # In FashionMNIST its features are in PIL image format, ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range [0., 1.]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # converts integers into one-hot encoded tensors
)

print(ds)