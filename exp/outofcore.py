import pandas
import numpy
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

data_dir = "../data/"
input_filename = data_dir + "HIGGS.csv"

# Read in 1 million examples at a time
num_examples = 11 * 10**6
block_size = 10**6



# Generate test set
df = pandas.read_csv(input_filename, nrows=5 * 10**5)
X_test = df.values[:, 1:]
y_test = df.values[:, 0]

# standardise data
scaler = StandardScaler()
df = pandas.read_csv(input_filename, skiprows=5 * 10**5, nrows=5 * 10**5)
X_train = df.values[:, 1:]
y_train = df.values[:, 0]

scaler.fit(X_train)
X_test = scaler.transform(X_test)

# Do some model selection on this dataset
metric = "accuracy"


learner = SGDClassifier(penalty="l1", n_jobs=-1, n_iter=10)
param_grid = [{"penalty": ["l1", "l2"], "alpha": 10.0**-numpy.arange(1, 7)}]
grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)

learner = MultinomialNB()
param_grid = [{"alpha": numpy.arange(0, 1, 0.1)}]
grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)

grid_search.fit(X_train, y_train)
learner = grid_search.best_estimator_
print(learner)


# Then do the learning
for i in range(10**6, num_examples, block_size):
    print(i)
    df = pandas.read_csv(input_filename, skiprows=i, nrows=block_size)

    X_train = df.values[:, 1:]
    y_train = df.values[:, 0]
    X_train = scaler.transform(X_train)
    learner.partial_fit(X_train, y_train, classes=numpy.array([0, 1]))

    try:
        y_pred_prob = learner.predict_proba(X_test)
    except:
        y_pred_prob = learner.decision_function(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(auc)
