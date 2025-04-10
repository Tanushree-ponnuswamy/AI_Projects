import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# === Parameters ===
data_path = 'data/train'  # Parent folder of "image" and "dummy"
model_save_path = 'model/autoencoder_skin_cancer.h5'
image_size = (128, 128)
batch_size = 16
epochs = 20

# === 1. Load only cancerous images ===
train_datagen = ImageDataGenerator(rescale=1./255)

# Only load data from 'image' subfolder (not 'dummy')
train_generator = train_datagen.flow_from_directory(
    directory=data_path,
    classes=['image'],  # Only the 'image' subfolder is included
    target_size=image_size,
    batch_size=batch_size,
    class_mode='input',  # Autoencoder learns to reconstruct its input
    shuffle=True
)

# === 2. Collect all training images into one NumPy array ===
train_images = []
for i in range(len(train_generator)):
    batch_x, _ = train_generator[i]
    train_images.append(batch_x)
    if len(train_images) * batch_size >= train_generator.samples:
        break
train_images = np.vstack(train_images)

# === 3. Define the Autoencoder Model ===
input_img = Input(shape=(128, 128, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# === 4. Train and Save the Model ===
checkpoint = ModelCheckpoint(model_save_path, monitor='loss', save_best_only=True)
autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

print("✅ Autoencoder training complete. Model saved at:", model_save_path)
