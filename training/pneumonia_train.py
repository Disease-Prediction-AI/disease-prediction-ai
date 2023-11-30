import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.optimizers.legacy import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint


data_dir = 'data/chest_xray'

train_dir = data_dir + '/train' # Path to train directory
val_dir = data_dir + '/val'
test_dir = data_dir + '/test' # Path to test directory


# Display Normal chest x-ray image
# img_normal = load_img('data/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
# print('NORMAL')
# plt.imshow(img_normal)
# plt.show()     


# Display Pneumonia chest x-ray image
# img_pneumonia = load_img('data/chest_xray/train/PNEUMONIA/person2_bacteria_4.jpeg')
# print('PNEUMONIA')
# plt.imshow(img_pneumonia)
# plt.show()

# Import VGG19 pre-trained model
# vgg_model = VGG19(include_top=True, weights='imagenet')
# vgg_model.summary()
vgg_model = VGG19(include_top=True, weights=None)

# Load the weights from your local file
weights_path = 'model/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
vgg_model.load_weights(weights_path)
vgg_model.summary()

# Pop off the last layer
vgg_model.layers.pop()

# Compile the model
predictions = Dense(1, activation='sigmoid')(vgg_model.layers[-1].output)
model = Model(inputs=vgg_model.input, outputs=predictions)

model.compile(optimizer = Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Prepare data augmentation configuration
train_datagen = ImageDataGenerator(rotation_range=40,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224,224),
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(val_dir,
                                                    target_size=(224,224),
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224,224),
                                                  batch_size=16,
                                                  class_mode='binary')


# Number of train and validation steps
train_steps=train_generator.n//train_generator.batch_size
validation_steps=validation_generator.n//validation_generator.batch_size

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_steps,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    epochs=10,
                   )

# Plot accuracy and loss graphs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Test the data
score = model.evaluate_generator(test_generator)

print('score = ', score)