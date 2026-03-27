import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Force synchronous execution
tf.config.run_functions_eagerly(True)

# ----------------------------
# Step 1: Extract dataset.zip
# ----------------------------
zip_file = "/content/dataset.zip"  # upload manually in Colab
extract_dir = "extracted_data"

with zipfile.ZipFile(zip_file, 'r') as z:
    z.extractall(extract_dir)
    print(f"Extracted zip to folder: {extract_dir}")

# ----------------------------
# Step 2: Data generators
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    extract_dir,
    target_size=(64,64),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    extract_dir,
    target_size=(64,64),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# ----------------------------
# Step 3: Build CNN
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 1 = blocked, 0 = good drainage
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Step 4: Train CNN
# ----------------------------
model.fit(train_gen, validation_data=val_gen, epochs=5)

# ----------------------------
# Step 5: Save model
# ----------------------------
model.save("cnn_drainage_model.h5")
print("Model trained and saved as cnn_drainage_model.h5")
