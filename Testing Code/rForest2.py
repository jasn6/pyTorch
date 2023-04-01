import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV

# Generate a random dataset for classification
X, y = make_classification(n_features=4, random_state=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4]
}

# Train a Random Forest classifier using cross-validation for hyperparameter tuning
rfc = RandomForestClassifier()
grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and mean cross-validation score
print('Best hyperparameters:', grid_search.best_params_)
print('Mean cross-validation score:', grid_search.best_score_)

# Evaluate the best model on the testing set
best_rfc = grid_search.best_estimator_
y_pred = best_rfc.predict(X_test)
accuracy = torch.sum(torch.tensor(y_pred == y_test)).item() / len(y_test)
print('Test set accuracy:', accuracy)
