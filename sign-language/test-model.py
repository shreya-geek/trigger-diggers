import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the model
model = tf.keras.models.load_model("sign_language_mnist_model.h5")

# Load the test data
test_df = pd.read_csv("sign_mnist_test.csv")
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Reshape and normalize the test data
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = X_test / 255.0

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")