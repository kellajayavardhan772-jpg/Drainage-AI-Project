import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from google.colab import files

tf.config.run_functions_eagerly(True)

# Load the pre-trained model (trained once)
model_file = "cnn_drainage_model.h5"
model = load_model(model_file)
print("Model loaded successfully! You can predict multiple images now.")

# Function to predict a single image
def predict_drainage(img_path):
    img = load_img(img_path, target_size=(64,64))
    x = img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0][0]
    confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100
    if pred > 0.5:
        return f"Blocked / High Risk – Drainage Exit May Be Problematic ({confidence:.2f}% confidence)"
    else:
        return f"Good Drainage – Exit is Better / Safe ({confidence:.2f}% confidence)"

# Upload new sample images
print("Upload sample images to predict:")
uploaded_img = files.upload()  # can upload multiple images

for img_path in uploaded_img.keys():
    result = predict_drainage(img_path)
    print(f"{img_path}: {result}")
