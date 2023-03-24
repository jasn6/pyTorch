import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

# Define the fields
TEXT = Field(tokenize='spacy', lower=True)
LABEL = LabelField(dtype=torch.float)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        pooled = torch.mean(embedded, dim=0)
        out = torch.relu(self.fc1(pooled))
        out = self.fc2(out)
        return out

# Define the hyperparameters
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Instantiate the model and move it to the GPU if available
model = SentimentClassifier(input_dim, embedding_dim, hidden_dim, output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Split the dataset into batches
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, device=device)

# Train the model
for epoch in range(num_epochs):
    for batch in train_iterator:
        optimizer.zero_grad()
        text = batch.text
        labels = batch.label
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on the test set
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in test_iterator:
            text = batch.text
            labels = batch.label
            output = model(text).squeeze(1)
            predictions = (output > 0).float()
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
        accuracy = total_correct / total_samples
    
    # Print the loss and accuracy
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, loss.item(), accuracy*100))
