import numpy
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import zero_one_loss

dataset = load_iris()

X = dataset["data"]
y = dataset["target"]

# Center each feature and scale the variance to be unitary
X = preprocessing.scale(X)

# Compute the variance for each column
print(numpy.var(X, 0).sum())

# Now use PCA using 3 components
pca = PCA(3)
X2 = pca.fit_transform(X)
print(numpy.var(X2, 0).sum())

pls = PLSRegression(3)
pls.fit(X, y)
X2 = pls.transform(X)
print(numpy.var(X2, 0).sum())

# Make predictions using an SVM with PCA and PLS
pca_error = 0
pls_error = 0
n_folds = 10

svc = LinearSVC()

for train_inds, test_inds in KFold(X.shape[0], n_folds=n_folds):
    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]

    # Use PCA and then classify using an SVM
    X_train2 = pca.fit_transform(X_train)
    X_test2 = pca.transform(X_test)

    svc.fit(X_train2, y_train)
    y_pred = svc.predict(X_test2)
    pca_error += zero_one_loss(y_test, y_pred)

    # Use PLS and then classify using an SVM
    X_train2, y_train2 = pls.fit_transform(X_train, y_train)
    X_test2 = pls.transform(X_test)

    svc.fit(X_train2, y_train)
    y_pred = svc.predict(X_test2)
    pls_error += zero_one_loss(y_test, y_pred)

print(pca_error / n_folds)
print(pls_error / n_folds)
