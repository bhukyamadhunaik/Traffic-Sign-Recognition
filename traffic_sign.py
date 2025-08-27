# traffic_signs.py

import os
from tensorflow.keras.layers import Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

# ================================
# Step 1: Load and preprocess data
# ================================
data, labels = [], []
classes = 43
cur_path = os.getcwd()

print("Loading training data...")
for i in range(classes):
    path_i = os.path.join(cur_path, 'dataset', 'train', str(i))
    images = os.listdir(path_i)
    for img_name in images:
        try:
            img = Image.open(os.path.join(path_i, img_name))
            img = img.resize((30, 30))
            data.append(np.array(img))
            labels.append(i)
        except Exception as e:
            print("Error loading image:", e)

data = np.array(data)
labels = np.array(labels)
print("Data shape:", data.shape, "Labels shape:", labels.shape)

# ================================
# Step 2: Train-test split
# ================================
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, classes)
y_val = to_categorical(y_val, classes)

# ================================
# Step 3: Build CNN model
# ================================
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(32, (5, 5), activation='relu'),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ================================
# Step 4: Train model
# ================================
epochs = 15
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs,
                    validation_data=(X_val, y_val))

# ================================
# Step 5: Plot accuracy & loss
# ================================
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.legend(), plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.legend(), plt.show()

# ================================
# Step 6: Save model
# ================================
model.save("traffic_classifier.h5")
print("Model saved as traffic_classifier.h5")

# ================================
# Step 7: Test dataset evaluation
# ================================
print("Evaluating on test dataset...")
y_test_df = pd.read_csv(os.path.join(cur_path, 'dataset', 'Test.csv'))
labels_test = y_test_df["ClassId"].values
img_paths = y_test_df["Path"].values

test_data = []
for path in img_paths:
    img = Image.open(os.path.join(cur_path, 'dataset', path))
    img = img.resize((30, 30))
    test_data.append(np.array(img))
X_test = np.array(test_data)

pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

print("Test Accuracy:", accuracy_score(labels_test, pred_labels))
