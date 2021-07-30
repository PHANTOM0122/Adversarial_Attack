########################################################################################################################
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

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
# default initial learning rate
learning_rate = 0.01
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return learning_rate * (drop_rate ** math.floor((1+epoch) / epochs_drop))
########################################################################################################################
'''
# Inputs and targets for KFold training set
inputs = X_train
targets = y_train
num_folds = 3

# Define K-fold cross validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold cross validation model evaluation
fold_no = 1

for train, test in kfold.split(inputs, targets):
    # Create and Train the Model for CIFAR-10
    model = Sequential()
    weight_decay = 0.0005
    x_shape = [32, 32, 3]

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # set model path
    Model_save_path = '/home/hj/PycharmProjects/HYPER/'
    if not os.path.exists(Model_save_path):
        os.mkdir(Model_save_path)
    Model_path = Model_save_path + 'fold%02d.hdf5'%fold_no

    # Checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=Model_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=True,
        verbose=1
    )

    # Generate Info for print
    print('------------------------------------------------------------------------------------------------------')
    print('Training fold:', fold_no)

    # Train model
    hist = model.fit(
        inputs[train], targets[train], epochs=1, batch_size=32, validation_data=(inputs[test], targets[test]),
        callbacks=[
            checkpoint, LearningRateScheduler(lr_step_decay, verbose=1)
        ]
    )

    # Generate genreralization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(model.metrics_names)
    print('Score for fold - loss', scores[0], 'accuracy:', scores[1])
    acc_per_fold.append(scores[1] * 100) # add % of accuracy
    loss_per_fold.append(scores[0]) # add loss

    # Save the entire model
    model.save('Saved_models%2d'%fold_no)

    fold_no += 1
    '''
########################################################################################################################
# without k-fold
# Create and Train the Model for CIFAR-10
model = Sequential()
weight_decay = 0.0005
x_shape = [32, 32, 3]

model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# set model path
Model_save_path = '/home/hj/PycharmProjects/HYPER/'
if not os.path.exists(Model_save_path):
    os.mkdir(Model_save_path)
Model_path = Model_save_path + 'VGG-16(cifar10).hdf5'

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
    X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test),
    callbacks=callbacks_list
)

model.save('VGG16_cifar10')
########################################################################################################################
# Adversarial Attack with foolbox
########################################################################################################################
from foolbox import TensorFlowModel, accuracy, samples, Model, utils, attacks, plot
from foolbox.attacks import LinfPGD, LinfDeepFoolAttack

model = tf.keras.models.load_model('VGG16_cifar10')
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

# clean_acc = accuracy(fmodel, attack_images, attack_labels)
# print(f"clean accuracy:  {clean_acc * 100:.1f} %")

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

# show images
foolbox.plot.images(clipped_advs[0], n=5, scale=3.) # clean
plt.show()

foolbox.plot.images(clipped_advs[4], n=5, scale=3.) # epsilon 0,01
plt.show()

foolbox.plot.images(clipped_advs[8], n=5, scale=3.) # epsilon 0,03
plt.show()

foolbox.plot.images(clipped_advs[9], n=5, scale=3.) # epsilon 0.1
plt.show()
