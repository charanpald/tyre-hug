from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)

sigma = 0.3

X, y, v = make_regression(100, 200, n_informative=150, noise=sigma, effective_rank=100, coef=True)
# X, y, v = make_regression(100, 50, n_informative=30, noise=sigma, effective_rank=20, coef=True)
# print(v)
# X, y = load_boston(return_X_y=True)
# X, y = load_diabetes(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)


X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

folds = 5
alphas = numpy.linspace(10.0**-3, 10, 200)
l1_ratios = 1 - numpy.logspace(-10, 0, 100)

# Ridge Regression
mses = numpy.zeros_like(alphas)

for i, alpha in enumerate(alphas):
    learner = Ridge(alpha=alpha, fit_intercept=True)
    scores = cross_val_score(learner, X, y, cv=folds, scoring="neg_mean_squared_error")
    mses[i] = numpy.abs(scores.mean())

learner = Ridge(alpha=alphas[numpy.argmin(mses)], fit_intercept=False)
learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Ridge Regression: alpha={:.4f} mse={:.3f} nnz={}".format(alphas[numpy.argmin(mses)], mse, numpy.count_nonzero(learner.coef_)))

# LASSO
mses = numpy.zeros_like(alphas)

learner = LassoCV(alphas=alphas, fit_intercept=False, cv=folds, n_jobs=-1)
learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("LASSO: alpha={:.4f} mse={:.3f} nnz={}".format(learner.alpha_, mse, numpy.count_nonzero(learner.coef_)))
print(y_pred)
print(learner.coef_)

# Elastic Net
learner = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, fit_intercept=False, cv=folds, n_jobs=-1, max_iter=5000)
learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Elastic Net: alpha={:.4f} l1_ratio={:.4f} mse={:.3f} nnz={}".format(learner.alpha_, learner.l1_ratio_, mse, numpy.count_nonzero(learner.coef_)))
