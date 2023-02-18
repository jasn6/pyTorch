import torch
from torch.utils.data import Dataset #  allow you to use pre-loaded datasets as well as your own data
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Dataset stores the samples and their corresponding labels
# Using FashionMNIST Dataset a dataset of Zalando’s article 
# images consisting of 60,000 training examples and 10,000 test examples. 
# Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

# Training_data
training_data = datasets.FashionMNIST(
    root="data",                            # where training/test data is stored
    train=True,                             # specifies training or test data
    download=True,                          # downloads the data from the internet if it’s not available at root
    transform=ToTensor()                    # specify the feature and label transformations
)

# Test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# Iterating and Visualizing the Dataset
# Can index Datasets manually like a list: training_data[index]
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # generate random index
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
