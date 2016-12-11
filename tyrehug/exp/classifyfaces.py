from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Input, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelBinarizer
import numpy
from keras.utils import np_utils

K.set_image_dim_ordering('th')

lfw_people = fetch_lfw_people(min_faces_per_person=30, resize=0.5)
images = lfw_people.images
images = numpy.reshape(images, (images.shape[0], 1, images.shape[1], images.shape[2]))

encoder = LabelBinarizer()
X = images.astype('float32')
X /= 255

print(X.shape)

# convert class vectors to binary class matrices
num_classes = numpy.unique(lfw_people.target).shape[0]
y = np_utils.to_categorical(lfw_people.target, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=lfw_people.target)

image_model = Sequential()
image_model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=X.shape[1:]))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 5, 5))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Flatten())
image_model.add(Dense(y.shape[1]))
image_model.add(Activation('softmax'))
image_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

image_model.fit(X_train, y_train, nb_epoch=10)

y_pred = image_model.predict(X_train)
# Pick largest response in each row
# y_pred = encoder.fit_transform(numpy.argmax(y_pred, 1))
# train_error = zero_one_loss(y_train, y_pred)
train_error = image_model.evaluate(X_train, y_train, verbose=0)

y_pred = image_model.predict(X_test)
# Pick largest response in each row
# y_pred = encoder.transform(numpy.argmax(y_pred, 1))
# test_error = zero_one_loss(y_test, y_pred)
test_error = image_model.evaluate(X_test, y_test, verbose=0)

print(train_error, test_error)
