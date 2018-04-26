import numpy
import os
from settings import DATA_DIR
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error


class RidgeRegression(object):

    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda

    def fit(self, X, y):
        C = X.T.dot(X) + self.lmbda * numpy.eye(X.shape[1])
        self.w = numpy.linalg.inv(C).dot(X.T.dot(y))

    def predict(self, X):
        return X.dot(self.w)

    def get_params(self, deep=True):
        return {"lmbda": self.lmbda}

    def set_params(self, lmbda=0.1):
        self.lmbda = lmbda
        return self


Xy = numpy.loadtxt(os.path.join(DATA_DIR, "winequality-white.csv"), delimiter=";", skiprows=1)

X = Xy[:, 0:-1]
X = scale(X)

y = Xy[:, -1]
y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


n_folds = 5

# Perform model selection then make predictions on the test set
ridge = RidgeRegression()
param_grid = [{"lmbda": 2.0**numpy.arange(-5, 10)}]
learner = GridSearchCV(ridge, param_grid, scoring="mean_absolute_error", n_jobs=-1, verbose=0)
learner.fit(X_train, y_train)

y_pred = learner.predict(X_test)
ridge_error = mean_absolute_error(y_test, y_pred)

# Perform model selection then make predictions on the test set
svr = SVR()
param_grid = [{"C": 2.0**numpy.arange(-5, 5)}]
learner = GridSearchCV(svr, param_grid, scoring="mean_absolute_error", n_jobs=-1, verbose=0)
learner.fit(X_train, y_train)

y_pred = learner.predict(X_test)
svc_error = mean_absolute_error(y_test, y_pred)


print("Ridge regression MAE     " + str(ridge_error))
print("SVC MAE                  " + str(svc_error))
