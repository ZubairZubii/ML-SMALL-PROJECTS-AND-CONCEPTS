import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load and preprocess images and masks
def load_data(image_dir, mask_dir, target_shape=(512, 512)):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)]
    
    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, target_shape) / 255.0
        images.append(image)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_shape)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Define U-Net model
def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Contracting Path
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)

    # Expansive Path
    up4 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)

    up5 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define paths
image_dir = r"C:\Users\Zubair\Documents\Javascript\ML\Gulocoma_detection_with_Unet model\Gulocoma_detection_with_Unet model\mask\fundus images"
mask_dir = r"C:\Users\Zubair\Documents\Javascript\ML\Gulocoma_detection_with_Unet model\Gulocoma_detection_with_Unet model\mask\mask"
# Load data and train
images, masks = load_data(image_dir, mask_dir)
model = unet_model(input_shape=(512, 512, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(images, masks, epochs=2, batch_size=4)
model.save('optic_disc_cup_segmentation_model.h5')
