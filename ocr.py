import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
data = pd.read_csv(url, header=None)

# Display first five records
print(data.head())
print("Dataset Shape:", data.shape)

# Input features
X = data.iloc[:, 1:].values
# Output labels
y = data.iloc[:, 0].values

# Convert letters into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Input Shape:", X.shape)
print("Output Shape:", y.shape)
print("Classes:", encoder.classes_)

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

#Normalise the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data Normalized Successfully!")

#Build the DNN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(16,)),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

model.summary()

#Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model Compiled Successfully!")

#Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

#Predict Letters
predictions = model.predict(X_test[:10])
predicted_labels = encoder.inverse_transform(
    np.argmax(predictions, axis=1)
)
actual_labels = encoder.inverse_transform(y_test[:10])
print("Predicted Letters:", predicted_labels)
print("Actual Letters: ", actual_labels)

#Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_
))

#Plot Training Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()