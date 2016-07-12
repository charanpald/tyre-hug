import numpy
import pandas
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from keras.models import Sequential
from keras.layers.core import Dense, Activation

dataset = load_digits()

X = dataset["data"]
y = dataset["target"]
y_indicators = pandas.get_dummies(y).values

# Center each feature and scale the variance to be unitary
X = preprocessing.scale(X)

svc = SVC(gamma=0.001)

# Set up variables
svc_error = 0
ann_error = 0
n_folds = 10


for train_inds, test_inds in KFold(X.shape[0], n_folds=n_folds):
    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]
    y_train_indicators, y_test_indicators = y_indicators[train_inds, :], y_indicators[test_inds, :]

    # Use the SVM
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    svc_error += zero_one_loss(y_test, y_pred)

    # Use deep learner
    ann = Sequential()
    ann.add(Dense(output_dim=256, input_dim=X.shape[1], init="glorot_uniform"))
    ann.add(Activation("relu"))
    ann.add(Dense(output_dim=10, init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    ann.compile(loss='categorical_crossentropy', optimizer='sgd')

    ann.fit(X_train, y_train_indicators, nb_epoch=50, batch_size=32)
    y_pred = ann.predict(X_test)
    # The predicted class is the output response with the largest value
    y_pred = numpy.argmax(y_pred, 1)
    ann_error += zero_one_loss(y_test, y_pred)

print(svc_error / n_folds)
print(ann_error / n_folds)
