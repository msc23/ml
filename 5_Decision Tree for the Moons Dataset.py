
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Generate the Moons dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2,random_state=42)


# Create an instance of the Decision Tree classifier
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
# e set the max_depth hyperparameter to 4, but you can experiment with differentvalues to see how it affects the accuracy of the model.
# Fit the model to the training data
tree_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = tree_clf.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
