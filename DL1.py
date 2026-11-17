import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import boston_housing

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print(f"Tensorflow version: {tf.__version__} ")

# Load the Dataset
(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

print(f"Training features shape : {X_train.shape}")
print(f"Training targets shape : {Y_train.shape}")
print(f"Testing features shape : {X_test.shape}")
print(f"Testing targets shape : {Y_test.shape}")

# Feature Normalization / Scaling
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

x_train_scaled = (X_train - mean) / std
x_test_scaled = (X_test - mean) / std

print(f"Feature Normalized successfully using Z-scored standardization")


# Build the Deep Neural Network Model
model = Sequential([
    Input(shape=(13,)),
    Dense(64, activation="relu", name="hidden_layer1"),
    Dense(32, activation="relu", name="hidden_layer2"),
    Dense(16, activation="relu", name="hidden_layer3"),
    Dense(1, activation="linear", name="output_layer")
])

model.summary()

# Compile the Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# Train the Model
EPOCH = 100
BATCH_SIZE = 16

history = model.fit(x_train_scaled, Y_train, batch_size=BATCH_SIZE,
                    epochs=EPOCH, verbose=1, validation_split=0.2)

# Evaluate Model on Test Set
test_mse, test_mae = model.evaluate(x_test_scaled, Y_test, verbose=0)

print(f"\n >> Final Test Performance: ")
print(f"Mean Squared Error (MSE) : {test_mse:.2f}")
print(f"Mean Absolute Error (MAE): ${test_mae * 1000:.2f}")

# Visualize Training Curves & Predictions
plt.figure(figsize=(14,5))

# Plot 1: Loss (MSE) Over Epochs
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training & Validation Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot 2: Actual vs Predicted Prices
y_pred = model.predict(x_test_scaled).flatten()
plt.subplot(1,2,2)
plt.scatter(Y_test, y_pred, color="blue", alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],'r--',lw=2)
plt.title("Actual vs Predceted Housing Prices")
plt.xlabel("Actual Price($1000s)")
plt.ylabel("Predicted Price($1000s)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Sample output comparison table for demonstration
sample_df = pd.DataFrame({
    'Actual Price': Y_test[:10],
    'Predicted Price': np.round(y_pred[:10],2),
    'Difference' : np.abs(Y_test[:10] - np.round(y_pred[:10],2))
})
print("\n--- Sample Predictions Output ---")
print(sample_df.to_string(index=False))