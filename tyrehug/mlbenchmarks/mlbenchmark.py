import numpy
import os
import sklearn.datasets as datasets
import sklearn.svm as svm
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
from tyrehug.settings import DATA_DIR, get_dir

learner = svm.SVC(kernel='linear', C=1)
data_dir = get_dir(os.path.join(DATA_DIR, "mlbenchmark"))
print(data_dir)

def benchmark(learner, dataset, filename):
    num_folds = 5
    num_metrics = 4
    scores = numpy.zeros((num_folds, num_metrics))

    X = preprocessing.scale(dataset.data)
    y = dataset.target

    num_labels = numpy.unique(y).shape[0]
    if num_labels == 2:
        average = "binary"
    else:
        average="weighted"

    if not os.path.exists(filename):
        for i, (train_inds, test_inds) in enumerate(cross_validation.StratifiedKFold(y, num_folds)):
            X_train, y_train = X[train_inds, :], y[train_inds]
            X_test, y_test = X[test_inds, :], y[test_inds]

            learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)

            scores[i, 0] = metrics.accuracy_score(y_test, y_pred)
            scores[i, 1] = metrics.f1_score(y_test, y_pred, average=average)
            scores[i, 2] = metrics.precision_score(y_test, y_pred, average=average)
            scores[i, 3] = metrics.recall_score(y_test, y_pred, average=average)

        numpy.save(filename, scores)
    else:
        scores = numpy.load(filename)

    print(scores.mean(0))
    print(scores.std(0))

wine_data = datasets.fetch_mldata('wine', data_home=data_dir)
ionosphere = datasets.fetch_mldata('ionosphere', data_home=data_dir, transpose_data=True)
diabetes = datasets.fetch_mldata('diabetes_scale', data_home=data_dir, transpose_data=True)
glass = datasets.fetch_mldata('glass', data_home=data_dir, transpose_data=True)
iris = datasets.load_iris()
digits = datasets.load_digits()

datasets_list = [("iris", iris), ("digits", digits), ("wine", wine_data), ("ionosphere", ionosphere), ("diabetes", diabetes), ("glass", glass)]

for dataset_name, dataset in datasets_list:
    print(dataset_name)
    print(numpy.unique(dataset.target))
    benchmark(learner, dataset, os.path.join(data_dir, "svm_" + dataset_name))

# TODO: Put in timings, memory, store in database for easy querying, visualise somehow
