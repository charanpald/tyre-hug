import pandas
import numpy
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn import pipeline
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.ensemble import RandomForestClassifier

data_dir = "../data/"
input_filename = data_dir + "HIGGS.csv"

num_examples = 11 * 10**6
num_test_examples = 5 * 10**5
num_train_examples = num_examples - num_test_examples
block_size = int(1.0 * 10**5)

# Generate test set (same as in paper)
df = pandas.read_csv(input_filename, skiprows=num_train_examples, nrows=num_test_examples)
X_test = df.values[:, 1:]
y_test = numpy.array(df.values[:, 0], numpy.int)

print(numpy.bincount(y_test))

# standardise data
scaler = StandardScaler()
df = pandas.read_csv(input_filename, skiprows=0, nrows=block_size)
X_train = df.values[:, 1:]
y_train = numpy.array(df.values[:, 0], numpy.int)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Do some model selection on this dataset
metric = "f1"


def smallscale():
    # Small scale learning
    learner_list = []

    gammas = 2.0**numpy.arange(-4, 6)

    feature_transformer = RBFSampler(gamma=.2, random_state=1)
    learner = PassiveAggressiveClassifier(n_iter=10)
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"learner__C": numpy.arange(0, 1, 0.2), "feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)


    feature_transformer = Nystroem(gamma=.2, random_state=1)
    learner = PassiveAggressiveClassifier(n_iter=10)
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"learner__C": numpy.arange(0, 1, 0.2), "feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)

    learner = PassiveAggressiveClassifier(n_iter=10)
    param_grid = [{"C": numpy.arange(0, 1, 0.2)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)


    feature_transformer = RBFSampler(gamma=.2, random_state=1)
    learner = SGDClassifier(n_iter=10, penalty="l1")
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"learner__alpha": 10.0**numpy.arange(-5, 2), "feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)


    feature_transformer = Nystroem(gamma=.2, random_state=1)
    learner = SGDClassifier(n_iter=10, penalty="l1")
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"learner__alpha": 10.0**numpy.arange(-5, 2), "feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)

    learner = SGDClassifier(n_iter=10, penalty="l1")
    param_grid = [{"alpha": 10.0**numpy.arange(-5, 2)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)

    feature_transformer = RBFSampler(gamma=.2, random_state=1)
    learner = GaussianNB()
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)

    feature_transformer = Nystroem(gamma=.2, random_state=1)
    learner = GaussianNB()
    learner_pipeline = pipeline.Pipeline([("feature_transformer", feature_transformer), ("learner", learner)])
    param_grid = [{"feature_transformer__gamma": gammas,
                    "feature_transformer__n_components": 2**numpy.arange(4, 8)}]
    grid_search = GridSearchCV(learner_pipeline, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    learner_list.append(grid_search)

    learner = GaussianNB()
    learner_list.append(learner)

    aucs = []

    for learner in learner_list:
        learner.fit(X_train, y_train)

        try:
            y_pred_prob = learner.predict_proba(X_test)[:, 1]
        except:
            y_pred_prob = learner.decision_function(X_test)

        auc = roc_auc_score(y_test, y_pred_prob)
        aucs.append(auc)
        print(auc)

    print(aucs)


def largescale2(X_train, y_train, X_test, y_test):
    # Feature generation using random forests
    feature_transformer = Nystroem(n_components=64, gamma=0.01)
    feature_transformer.fit(numpy.zeros((1, X_train.shape[1])))

    grid_search_list = []
    learner_list = []

    # Do some model selection on the model selection set
    learner = PassiveAggressiveClassifier(n_iter=1)
    param_grid = [{"C": numpy.arange(0, 1.5, 0.1)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    grid_search_list.append(grid_search)

    learner = Perceptron(n_iter=1, penalty="elasticnet")
    param_grid = [{"alpha": 10.0**numpy.arange(-5, 0)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    grid_search_list.append(grid_search)

    learner = SGDClassifier(n_iter=1, penalty="elasticnet")
    param_grid = [{"alpha": 10.0**numpy.arange(-5, 0), "l1_ratio": numpy.linspace(0, 1, 10)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    grid_search_list.append(grid_search)

    # Include Naive Bayes
    learner_list.append(MultinomialNB())
    learner_list.append(BernoulliNB())


    for grid_search in grid_search_list:
        X_train2 = feature_transformer.transform(X_train)
        grid_search.fit(X_train2, y_train)
        learner_list.append(grid_search.best_estimator_)

    aucs = []

    # Then do the learning
    # TODO: Check we don't infringe on the test set
    for i in range(block_size, num_train_examples, block_size):
        print(i)
        df = pandas.read_csv(input_filename, skiprows=i, nrows=block_size)

        X_train = df.values[:, 1:]
        y_train = numpy.array(df.values[:, 0], numpy.int)
        X_train = scaler.transform(X_train)

        auc_row = []

        for learner in learner_list:
            X_train2 = feature_transformer.transform(X_train)
            learner.partial_fit(X_train2, y_train, classes=numpy.array([0, 1]))

            X_test2 = feature_transformer.transform(X_test)

            try:
                y_pred_prob = learner.predict_proba(X_test2)[:, 1]
            except:
                y_pred_prob = learner.decision_function(X_test2)

            auc = roc_auc_score(y_test, y_pred_prob)
            auc_row.append(auc)
            print(auc)

        aucs.append(auc_row)

    print(aucs)
    return numpy.array(aucs)

def largescale(X_train, y_train, X_test, y_test):
    # Feature generation using random forests
    forest = RandomForestClassifier(n_estimators=128, n_jobs=-1)
    forest.fit(X_train, y_train)
    encoder = OneHotEncoder()
    encoder.fit(forest.apply(X_train))

    grid_search_list = []
    learner_list = []

    # Do some model selection on the model selection set
    # learner = PassiveAggressiveClassifier(n_iter=1)
    # param_grid = [{"C": numpy.arange(0, 1.5, 0.1)}]
    # grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    # grid_search_list.append(grid_search)

    # learner = Perceptron(n_iter=1, penalty="elasticnet")
    # param_grid = [{"alpha": 10.0**numpy.arange(-5, 0)}]
    # grid_search = GridSearchCV(learner, param_grid, n_jobs=-1, verbose=2, scoring=metric, refit=True)
    # grid_search_list.append(grid_search)
    #
    # Try different learning rates
    learner = SGDClassifier(n_iter=5, penalty="elasticnet", learning_rate="optimal", eta0=0.1)
    param_grid = [{"alpha": 10.0**numpy.arange(-5, 0), "l1_ratio": numpy.linspace(0, 1, 10)}]
    grid_search = GridSearchCV(learner, param_grid, n_jobs=2, verbose=2, scoring=metric, refit=True)
    grid_search_list.append(grid_search)
    #
    # # Include Naive Bayes
    # learner_list.append(MultinomialNB())
    # learner_list.append(BernoulliNB())


    for grid_search in grid_search_list:
        X_train2 = encoder.transform(forest.apply(X_train))
        grid_search.fit(X_train2, y_train)
        learner_list.append(grid_search.best_estimator_)

    print(learner_list)

    aucs = []

    # Then do the learning
    # TODO: Check we don't infringe on the test set
    for i in range(block_size, num_train_examples, block_size):
        print(i)
        df = pandas.read_csv(input_filename, skiprows=i, nrows=block_size)

        X_train = df.values[:, 1:]
        y_train = numpy.array(df.values[:, 0], numpy.int)
        X_train = scaler.transform(X_train)

        auc_row = []

        for learner in learner_list:
            X_train2 = encoder.transform(forest.apply(X_train))
            learner.partial_fit(X_train2, y_train, classes=numpy.array([0, 1]))

            X_test2 = encoder.transform(forest.apply(X_test))

            try:
                y_pred_prob = learner.predict_proba(X_test2)[:, 1]
            except:
                y_pred_prob = learner.decision_function(X_test2)

            auc = roc_auc_score(y_test, y_pred_prob)
            auc_row.append(auc)
            print(auc)

        aucs.append(auc_row)

    print(aucs)
    return numpy.array(aucs)

aucs = largescale(X_train, y_train, X_test, y_test)
