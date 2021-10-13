import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import os
import json
import numpy as np
from keras.models import model_from_json
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

img_size = 256

train_path = r"E:\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
valid_path = (r"E:\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid")
training_images = []

for root, dirs, files in os.walk(train_path):
    for file in files:
        training_images.append(os.path.join(root, file))

validation_images = []

for root, dirs, files in os.walk(valid_path):
    for file in files:
        validation_images.append(os.path.join(root, file))

print("Training:")
print("Training Path:" + train_path)
print("Training Classes:" + str(len(os.listdir(train_path))))
print("Training Images:" + str(len(training_images)))

print("\n")

print("Validation:")
print("Validation Path:" + valid_path)
print("Validation Classes:" + str(len(os.listdir(valid_path))))
print("Validation Images:" + str(len(validation_images)))

train_datagen = ImageDataGenerator(
    rescale=1/255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1/255)

batch_size = 32


train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size = batch_size,
    class_mode = 'categorical',
    target_size = (150, 150),
    color_mode="rgb",
    shuffle=True
)

validation_generator =  validation_datagen.flow_from_directory(
    valid_path,
    batch_size  = batch_size,
    class_mode  = 'categorical',
    target_size = (150, 150),
    color_mode="rgb",
    shuffle=True
)

class_dict = train_generator.class_indices
train_num = train_generator.samples
valid_num = validation_generator.samples

inception_model = InceptionV3(input_shape= (150, 150, 3),
                                include_top = False,
                                weights = None)

# inception_model.load_weights(local_weights_file)
# Freezing all Layers of Inception V3 Model
for layer in inception_model.layers:
    layer.trainable = False

inception_model.summary()

# Taking output from 'mixed8' layer
last_layer = inception_model.get_layer('mixed9')
print('Last Layer Output Shape:', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(2048, activation='relu')(x)

# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(len(class_dict), activation='softmax')(x)

model = Model(inception_model.input, x)

model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_num//batch_size,
    validation_data = validation_generator,
    validation_steps=valid_num//batch_size,
    epochs = 5,
#     callbacks=callbacks_list,
    verbose = 1
)

model.evaluate_generator(
    generator=validation_generator,
    steps=valid_num//batch_size
)
