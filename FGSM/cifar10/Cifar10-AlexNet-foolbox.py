########################################################################################################################
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential

import sklearn
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import time
import math
import eagerpy as ep

import foolbox
########################################################################################################################
# Load cifar-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Define score containers for each fold
acc_per_fold = []
loss_per_fold = []
########################################################################################################################
# Normalize
X_train = (X_train / 255.0)
X_test = (X_test / 255.0)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
########################################################################################################################
# show images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

########################################################################################################################
# AlexNet
########################################################################################################################
# without k-fold
# Create and Train the Model for CIFAR-10
class AlexNet8(Sequential):

    def __init__(self, input_shape, num_classes):
        super().__init__()

        # 1st Block(Input + Conv2D + MaxPooling + Normalization)
        self.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 2nd Block(Input + Conv2D + MaxPooling + Normalization)
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd Block(Conv2D)
        self.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # FC layer
        self.add(Flatten())  # 2D -> 1D vectors
        self.add(Dense(2048, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(2048, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))
img_height = 32
img_width = 32
img_channels = 3
input_shape = (img_height, img_width, img_channels)
num_classes = 10

# Build model
model = AlexNet8(input_shape=input_shape, num_classes=num_classes)

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# set model path
Model_save_path = '/home/hj/PycharmProjects/HYPER/'
if not os.path.exists(Model_save_path):
    os.mkdir(Model_save_path)
Model_path = Model_save_path + 'AlexNet(cifar10).hdf5'

patient = 3
callbacks_list=[
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.1,
        patience=patient,
        min_lr=0.00001,
        verbose=1,
        mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=Model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )]

# Train model
hist = model.fit(
    X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test),
    callbacks=callbacks_list
)

model.save('AlexNet_cifar10')
########################################################################################################################
# Adversarial Attack with foolbox
########################################################################################################################
from foolbox import TensorFlowModel, accuracy, samples, Model, utils, attacks, plot
from foolbox.attacks import LinfPGD, LinfDeepFoolAttack

model = tf.keras.models.load_model('AlexNet_cifar10')
model.load_weights(Model_path)
preprocessing = dict()
bounds = (0, 1)
fmodel = TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
fmodel = fmodel.transform_bounds((0, 1))

attack_labels = tf.convert_to_tensor(y_test, dtype='int64')
attack_labels = tf.reshape(attack_labels, 10000)
attack_images = tf.convert_to_tensor(X_test, dtype='float32')

predictions = model.predict(attack_images)
orig_predictions = np.argmax(predictions, axis=1)

########################################################################################################################
attack_images = ep.astensor(attack_images)
attack_labels = ep.astensor(attack_labels)


# apply the attack
attack = foolbox.attacks.FGSM()
epsilons = [
    0.0,
    0.0002,
    0.0005,
    0.0008,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.01,
    0.1,
    0.3,
    0.5,
    1.0,
]

raw_advs, clipped_advs, success = attack(fmodel, attack_images[:20], attack_labels[:20], epsilons=epsilons)

# calculate and report the robust accuracy (the accuracy of the model when
# it is attacked)
robust_accuracy = 1 - success.float32().mean(axis=-1)
print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons, robust_accuracy):
    print(f"  epsilon = {eps:<6}: {acc.item() * 100:4.1f} %")

print('Attack finshed!')

'''
# print(len(raw_advs)) # 13
# print(raw_advs[0]) # shpae : 20, 32, 32, 3 dtype = float32
# print(len(clipped_advs)) # 13
# print(clipped_advs[0]) # shape : 20, 32, 32, 3, dtype = float32

print(len(raw_advs))
print(len(raw_advs[0]))
print(len(raw_advs[0][0]))
print(len(raw_advs[0][0][0]))

plt.figure()
plt.imshow(raw_advs[0][0])
plt.show()


raw_advs = np.array(raw_advs, dtype='uint8')
print('converted done!')

# show images
plt.figure(figsize=(10,10))
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(raw_advs[0][0])
plt.show()
'''


