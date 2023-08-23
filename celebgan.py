#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image_path = r"C:\Users\Amylicious\Downloads\img_align_celeba\img_align_celeba\000001.jpg"
image = cv2.imread(input_image_path)
height, width = image.shape[:2] # getting the image size
k1 = 0.5
k2 = 0.2

def fish_eye_distortion(image, k1, k2):
 height, width, _ = image.shape

 # Create an x, y coordinate grid for the image
 x = np.linspace(-1, 1, width)
 y = np.linspace(-1, 1, height)
 x, y = np.meshgrid(x, y)

 # r is the distance from the center
 r = np.sqrt(x ** 2 + y ** 2)

 # Apply the fisheye effect, based on the radial distance from the center
 distorted_radius = r + k1 * r ** 3 + k2 * r ** 5

 # Get the distorted x, y coordinates
 x_distorted = distorted_radius * x / r
 y_distorted = distorted_radius * y / r

 # Map from distorted coordinates to original image coordinates
 x_map = ((x_distorted + 1) * width) / 2
 y_map = ((y_distorted + 1) * height) / 2

 # Interpolate using remap with CONSTANT border mode and white as border value
 distorted_image = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32),
 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])

 return distorted_image


if image is None:
 print("Error loading image.")
else:
 height, width = image.shape[:2]

if __name__ == "__main__":
 distorted_image = fish_eye_distortion(image, k1, k2)

# Convert both original and distorted images to RGB for displaying using matplotlib
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(distorted_image)
plt.title("Distorted Image")
plt.axis('off')

plt.tight_layout()
plt.show()
  


# In[4]:


# create directories for distorted images 
import os

distorted_dir = '/kaggle/working/distorted'
if not os.path.exists(distorted_dir):
    os.mkdir(distorted_dir)


# In[8]:


# distort and save images
import PIL

source_dir = '/kaggle/input/celebahq-resized-256x256/celeba_hq_256'
dest_dir = '/kaggle/working/distorted'
for filename in os.listdir(source_dir):
    input_image_path = os.path.join(source_dir, filename)
    image = cv2.imread(input_image_path)
    distorted_image = fish_eye_distortion(image, k1, k2)
    dest_file_path = os.path.join(dest_dir, filename)
    # Convert the NumPy array to a PIL image
    img = PIL.Image.fromarray(distorted_image)

    # Save the image
    img.save(dest_file_path,format='jpeg')    


# In[9]:


# pair images
original_dir = source_dir
distorted_dir = dest_dir

image_pairs = []

original_files = sorted(os.listdir(original_dir))
distorted_files = sorted(os.listdir(distorted_dir))

for orig, dist in zip(original_files, distorted_files):
    image_pairs.append((os.path.join(distorted_dir, dist), os.path.join(original_dir, orig)))


# In[10]:


# split dataset
from sklearn.model_selection import train_test_split

train_pairs, temp_pairs = train_test_split(image_pairs, test_size=0.2, random_state=42)
valid_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)


# In[12]:


import tensorflow as tf

# load data 
def load_image(image_pair):
    # Split the image_pair tensor into distorted_path and original_path
    distorted_path = image_pair[0]
    original_path = image_pair[1]
    
    distorted_image = tf.image.decode_jpeg(tf.io.read_file(distorted_path))
    original_image = tf.image.decode_jpeg(tf.io.read_file(original_path))
    
    distorted_image = tf.image.resize(distorted_image, [256, 256])
    original_image = tf.image.resize(original_image, [256, 256])
        
    return distorted_image, original_image


# In[13]:


# normalize data
def normalize_image(image):
    return (image / 127.5) - 1  # Normalize to [-1, 1]

def load_and_normalize_image(image_pair):
    distorted, original = load_image(image_pair)
    return normalize_image(distorted), normalize_image(original)

train_dataset = tf.data.Dataset.from_tensor_slices(train_pairs)
train_dataset = train_dataset.map(load_and_normalize_image).batch(32).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[15]:


# define generator
from tensorflow.keras import layers

def build_generator(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(x)  # 3 channels for RGB

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# In[16]:


# define discriminator
def build_discriminator(input_shape):
    distorted_input = layers.Input(shape=input_shape)
    clean_input = layers.Input(shape=input_shape)

    merged = layers.Concatenate(axis=-1)([distorted_input, clean_input])

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(merged)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[distorted_input, clean_input], outputs=outputs)


# In[17]:


# build generator and discriminator
generator = build_generator(input_shape=(256, 256, 3))
discriminator = build_discriminator(input_shape=(256, 256, 3))

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[18]:


# define discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# define generator loss
def generator_loss(fake_output, generated_images, target_images):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    # Mean Squared Error as the reconstruction loss
    l2_loss = tf.reduce_mean(tf.square(generated_images - target_images))
    total_loss = gan_loss + (100 * l2_loss)  # The weight for l2_loss can be adjusted
    return total_loss


# In[20]:


# define training step
@tf.function
def train_step(distorted_images, clean_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(distorted_images, training=True)

        real_output = discriminator([distorted_images, clean_images], training=True)
        fake_output = discriminator([distorted_images, generated_images], training=True)

        gen_loss = generator_loss(fake_output, generated_images, clean_images)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[21]:


import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Select a few image paths from the validation set
sample_image_paths = random.sample([pair[0] for pair in valid_pairs],16)

# Load and preprocess the images
sample_distorted_images = [img_to_array(load_img(img_path, target_size=(256, 256))) for img_path in sample_image_paths]
sample_distorted_images = np.array(sample_distorted_images)
sample_distorted_images = (sample_distorted_images - 127.5) / 127.5  # Normalize to [-1, 1]


# In[22]:


# define training loop
def generate_and_display_images(model, test_input):
    # Generate images from the model
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
        plt.axis('off')

    plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        for distorted_batch, clean_batch in dataset:
            train_step(distorted_batch, clean_batch)
            
        # Generate and display images at the end of the epoch
        generate_and_display_images(generator, sample_distorted_images)


# In[ ]:


EPOCHS = 100
train(train_dataset, EPOCHS)

