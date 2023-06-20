import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# Load the Auto MPG dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t',sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()
# Split the dataset into a training set and a test set
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# Separate the target variable, "MPG", from the input features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
# Normalize the input features
train_stats = train_dataset.describe().transpose()
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# Define the model architecture
model = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=[len(train_dataset.
keys())]),
layers.Dense(64, activation='relu'),
layers.Dense(1)
])
# Compile the model
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
# Train the model
EPOCHS = 1000
history = model.fit(
normed_train_data, train_labels,
epochs=EPOCHS, validation_split = 0.2, verbose=0)
# Evaluate the model on the test set
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
# Make predictions on new data
test_predictions = model.predict(normed_test_data).flatten()
# Plot predicted vs actual values

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 100], [0, 100], color='red')
plt.show()