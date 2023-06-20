import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the iris dataset
iris = load_iris()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
test_size=0.2, random_state=42)
# Add bias term to the feature matrices
X_train_bias = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_bias = np.c_[np.ones((len(X_test), 1)), X_test]
# Define the softmax function
def softmax(logits): 
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    return exp_logits / sum_exp_logits
# Initialize the model parameters
n_inputs = X_train_bias.shape[1]
n_outputs = len(np.unique(y_train))
theta = np.random.randn(n_inputs, n_outputs)
# Define the learning rate, number of epochs, and early stopping parameters
eta = 0.1
n_epochs = 1000
max_epochs_without_improvement = 50
best_loss = np.inf
epochs_without_improvement = 0
# Train the model using batch gradient descent with early stopping
for epoch in range(n_epochs):
# Compute the logits and probabilities for the training set
    logits_train = X_train_bias.dot(theta)
    y_proba_train = softmax(logits_train)
# Compute the loss and gradient for the training set
    loss_train = -np.mean(np.sum(np.log(y_proba_train) * (y_train.reshape(-1,1) == np.arange(n_outputs)), axis=1))
    error_train = y_proba_train - (y_train.reshape(-1, 1) == np.arange(n_outputs))
    grad = 1/len(X_train_bias) * X_train_bias.T.dot(error_train)
# Update the model parameters
    theta -= eta * grad
# Compute the logits and probabilities for the testing set
    logits_test = X_test_bias.dot(theta)
    y_proba_test = softmax(logits_test)
# Compute the loss for the testing set
    loss_test = -np.mean(np.sum(np.log(y_proba_test) * (y_test.reshape(-1, 1)== np.arange(n_outputs)), axis=1))
# Check if the loss on the testing set has improved
    if loss_test < best_loss:
        best_loss = loss_test
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
# Print the loss every 50 epochs
    if epoch % 50 == 0:
        print("Epoch:", epoch, "Loss(train):", loss_train, "Loss(test):",loss_test)
# Check if the early stopping criteria have been met
    if epochs_without_improvement > max_epochs_without_improvement:
        print("Early stopping!")
        break
# Make predictions on the testing set using the trained model
logits_test = X_test_bias.dot(theta)
y_proba_test = softmax(logits_test)
y_pred_test = np.argmax(y_proba_test, axis=1)
# Compute the accuracy of the model on the testing set
accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy:", accuracy)