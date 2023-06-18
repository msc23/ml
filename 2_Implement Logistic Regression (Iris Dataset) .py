import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()

# Split the data into features and target
X = iris.data
y = iris.target

# Create a logistic regression model
clf = linear_model.LogisticRegression()

# Train the model
clf.fit(X, y)

# Make predictions
y_pred = clf.predict(X)

# Calculate the accuracy
accuracy = accuracy_score(y, y_pred)

print('Accuracy:', accuracy)

# Plot the results
plt.scatter(X[:, 0], y, color='black')
plt.plot(X[:, 0], y_pred, color='red')
plt.xlabel('Sepal length')
plt.ylabel('Species')
plt.show()