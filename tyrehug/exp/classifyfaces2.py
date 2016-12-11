from __future__ import print_function
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils import np_utils
from keras import backend as K
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from keras.regularizers import activity_l2, l2

numpy.random.seed(21)
batch_size = 10
nb_epoch = 50

# TODO: See how they do the preprocessing

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
lfw_people = fetch_lfw_people(min_faces_per_person=30, resize=1.0)
X = lfw_people.images
y = lfw_people.target
img_rows, img_cols = X.shape[1:]
nb_classes = numpy.unique(y).shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=lfw_people.target)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
"""
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(BatchNormalization(axis=1, mode=2))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(BatchNormalization(axis=1, mode=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization(axis=1, mode=2))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
"""

inputs = Input(shape=X_train.shape[1:])
conv1 = BatchNormalization(axis=1, mode=2)(Convolution2D(8, 7, 7, activation='relu', border_mode='same')(inputs))
conv2 = BatchNormalization(axis=1, mode=2)(Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1))
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
dropout1 = Dropout(0.2)(pool1)

conv3 = BatchNormalization(axis=1, mode=2)(Convolution2D(32, 3, 3, activation='relu', border_mode='same')(dropout1))
conv4 = BatchNormalization(axis=1, mode=2)(Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3))
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
dropout2 = Dropout(0.2)(pool2)

conv8 = BatchNormalization(axis=1, mode=2)(Convolution2D(32, 7, 7, activation='relu', border_mode='same')(dropout2))
up8 = UpSampling2D(size=(2, 2))(conv8)

conv9 = BatchNormalization(axis=1, mode=2)(Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8))
conv10 = BatchNormalization(axis=1, mode=2)(Convolution2D(8, 7, 7, activation='relu', border_mode='same')(conv9))

flatten = Flatten()(conv10)
dropout1 = Dropout(0.2)(flatten)
dense1 = Dense(nb_classes, activation="softmax")(dropout1)
model = Model(input=inputs, output=dense1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
