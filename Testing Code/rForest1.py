import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a random dataset for classification
X, y = make_classification(n_features=4, random_state=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = rfc.predict(X_test)
accuracy = torch.sum(torch.tensor(y_pred == y_test)).item() / len(y_test)
print('Accuracy:', accuracy)
