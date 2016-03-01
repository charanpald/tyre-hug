import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def loaddata(input_filename, num_test_examples, block_size=10**5):
    # Generate test set as the last 500,000 rows (same as in paper)
    df = pandas.read_csv(input_filename, header=None, skiprows=num_train_examples, nrows=num_test_examples)
    X_test = df.values[:, 1:]
    y_test = numpy.array(df.values[:, 0], numpy.int)

    # Training data
    df = pandas.read_csv(input_filename, header=None, skiprows=0, nrows=block_size)
    X_train = df.values[:, 1:]
    y_train = numpy.array(df.values[:, 0], numpy.int)

    # standardise data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def modelselect(input_filename, num_test_examples, block_size, n_estimators=100):
    # Perform some model selection to determine good parameters
    # Load data
    X_train, y_train, X_test, y_test, scaler = loaddata(input_filename, num_test_examples, block_size)

    # Feature generation using random forests
    forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    forest.fit(X_train, y_train)
    encoder = OneHotEncoder()
    encoder.fit(forest.apply(X_train))
    X_train = encoder.transform(forest.apply(X_train))
    learner = SGDClassifier(loss="hinge", penalty="l2", learning_rate="invscaling", alpha=0.001, average=10**4, eta0=0.5, class_weight="balanced")

    metric = "f1"
    losses = ["log", "hinge", "modified_huber", "squared_hinge", "perceptron"]
    penalties = ["l2", "l1", "elasticnet"]
    alphas = 10.0**numpy.arange(-5, 0)
    learning_rates = ["constant", "optimal", "invscaling"]
    param_grid = [{"alpha": alphas, "loss": losses, "penalty": penalties, "learning_rate": learning_rates}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_, grid_search.best_score_)
    return grid_search


def train(input_filename, num_train_examples, num_test_examples, block_size):
    # Load initial training data and test data
    X_train, y_train, X_test, y_test, scaler = loaddata(input_filename, num_test_examples, block_size)

    # Feature generation using random forests
    forest = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    forest.fit(X_train, y_train)
    encoder = OneHotEncoder()
    encoder.fit(forest.apply(X_train))
    X_test = encoder.transform(forest.apply(X_test))
    # Make sure that classes are weighted inversely to their frequencies
    weights = float(y_train.shape[0]) / (2 * numpy.bincount(y_train))
    class_weights = {0: weights[0], 1: weights[1]}
    learner = SGDClassifier(loss="hinge", penalty="l2", learning_rate="invscaling", alpha=0.0001, average=10**4, eta0=1.0,
                class_weight=class_weights)

    num_passes = 3
    aucs = []

    for j in range(num_passes):
        for i in range(0, num_train_examples, block_size):
            df = pandas.read_csv(input_filename, header=None, skiprows=i, nrows=block_size)
            X_train = df.values[:, 1:]
            X_train = scaler.transform(X_train)
            X_train = encoder.transform(forest.apply(X_train))
            y_train = numpy.array(df.values[:, 0], numpy.int)
            del df

            learner.partial_fit(X_train, y_train, classes=numpy.array([0, 1]))
            y_pred_prob = learner.decision_function(X_test)
            auc = roc_auc_score(y_test, y_pred_prob)
            aucs.append([i + num_train_examples * j, auc])
            print(aucs[-1])

    df = pandas.DataFrame(aucs, columns=["Iterations", "AUC"])
    df = df.set_index("Iterations")
    return df

block_size = 10**5
num_examples = 11 * 10**6
num_test_examples = 5 * 10**5
num_train_examples = num_examples - num_test_examples
data_dir = "../data/"
input_filename = data_dir + "HIGGS.csv"

aucs = train(input_filename, num_train_examples, num_test_examples, block_size)
print(aucs)
aucs.to_csv("aucs.csv")

aucs.plot(legend=False)
