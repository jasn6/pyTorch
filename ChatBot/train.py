import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ntlk_utils import stem, tokenize, bag_of_words

with open('intents.json','r') as f:
    intents = json.load(f)

# create a list of all pattern words
all_words = []
tags = []
# a list of both pattern words and tags
xy = []
for intent in intents['intents']:
    tags.append(intent["tag"])
    for pattern in intent['patterns']:
        # can add on to all_words by tokenizing each phrase
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,intent["tag"]))

ignore_words = ['?','!','.',',']
# stem the words
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
batch_size = 8

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)