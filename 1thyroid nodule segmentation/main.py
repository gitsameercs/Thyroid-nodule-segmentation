from __future__ import print_function
import numpy as np
import pandas as pd
import os
import logging  # Import logging for error handling
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Define functions for image and mask preprocessing
def preprocess_image(image_path):
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalize pixel values to the range [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading image from {image_path}: {str(e)}")
        return None

def preprocess_mask(mask_path):
    try:
        img = load_img(mask_path, color_mode='grayscale', target_size=(128, 128))
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalize pixel values to the range [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading mask from {mask_path}: {str(e)}")
        return None

# Define a custom data generator
def data_generator(images, masks, batch_size, image_size):
    while True:
        batch_image = []
        batch_mask = []
        for i in range(batch_size):
            # Randomly select an image and its corresponding mask
            idx = np.random.randint(0, len(images))
            image_path = images[idx]
            mask_path = masks[idx]
            
            # Load and preprocess the image and mask
            image = preprocess_image(image_path)
            mask = preprocess_mask(mask_path)
            
            if image is not None and mask is not None:
                batch_image.append(image)
                batch_mask.append(mask)
        
        yield np.array(batch_image), np.array(batch_mask)

def unet(height=None, width=None, channels=1, features=32, depth=4, padding='same', dropout=0.0):
    inputs = tf.keras.layers.Input((height, width, channels))
    skip_connections = []

    # Contracting path
    for _ in range(depth):
        conv1 = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(inputs)
        conv2 = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(conv1)
        skip_connections.append(conv2)
        pool = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        features *= 2
        dropout_layer = tf.keras.layers.Dropout(dropout)(pool) if dropout > 0 else pool

    # Bottom of the U-Net
    conv_bottom = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(dropout_layer)
    conv_bottom = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(conv_bottom)

    # Expanding path
    for conv in reversed(skip_connections):
        features //= 2
        upsample = tf.keras.layers.UpSampling2D((2, 2))(conv_bottom)
        
        # Ensure the spatial dimensions match by resizing the 'upsample' tensor
        upsample = tf.image.resize(upsample, (conv.shape[1], conv.shape[2]), method=tf.image.ResizeMethod.BILINEAR)
        
        # Concatenate the tensors with matching spatial dimensions
        concat = tf.keras.layers.Concatenate(axis=-1)([conv, upsample])
        conv1 = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(concat)
        conv2 = tf.keras.layers.Conv2D(features, (3, 3), activation='relu', padding=padding)(conv1)
        conv_bottom = conv2

    # Output layer
    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_bottom)

    return tf.keras.Model(inputs=inputs, outputs=output)

# Load your dataset and preprocess it
image_dir = 'C:/Users/MY-PC/Desktop/ravi project/thyroid-nodule-segmentation-main/dataset/image'
mask_dir = 'C:/Users/MY-PC/Desktop/ravi project/thyroid-nodule-segmentation-main/dataset/mask'

image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]

# Split the dataset into training and validation sets
image_train, image_test, mask_train, mask_test = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

# Load the U-Net model
model = unet(height=128, width=128, channels=1)  # Modify the input dimensions as needed

# Compile the model
model.compile(optimizer=Adam(learning_rate =1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Create checkpoints to save the model during training
checkpoint = ModelCheckpoint('segmentation_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train the model
batch_size = 8
num_epochs = 50

# Use a separate validation dataset for monitoring accuracy during training
image_train, image_val, mask_train, mask_val = train_test_split(image_train, mask_train, test_size=0.2, random_state=42)
train_data_generator = data_generator(image_train, mask_train, batch_size, (128, 128))
val_data_generator = data_generator(image_val, mask_val, batch_size, (128, 128))

history = model.fit(train_data_generator,
                    steps_per_epoch=len(image_train) // batch_size,
                    epochs=num_epochs,
                    validation_data=val_data_generator,
                    validation_steps=len(image_val) // batch_size,
                    callbacks=[checkpoint])

# Evaluate the model on the test dataset and calculate accuracy
test_data_generator = data_generator(image_test, mask_test, batch_size, (128, 128))
y_true = []
y_pred = []

for _ in range(len(image_test) // batch_size):
    images, masks = next(test_data_generator)
    predictions = model.predict(images)
    
    # Flatten the masks and predictions for accuracy calculation
    masks = masks.reshape(-1)
    predictions = predictions.reshape(-1)
    
    y_true.extend(masks)
    y_pred.extend(predictions)

# Calculate accuracy using a threshold
threshold = 0.5
y_pred_binary = np.array([1 if val > threshold else 0 for val in y_pred])
y_true = np.array(y_true)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_binary)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Save the final trained model
model.save('final_segmentation_model.h5')

# To load the trained model for prediction, use the following:
# loaded_model = load_model('final_segmentation_model.h5')

# Use the loaded model for segmentation predictions on new data
# You can load new ultrasound images, preprocess them, and then use the model to predict segmentation masks
