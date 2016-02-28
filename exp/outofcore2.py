import pandas
import numpy
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

data_dir = "../data/"
input_filename = data_dir + "HIGGS.csv"
num_examples = 11 * 10**6
num_test_examples = 5 * 10**5
num_train_examples = num_examples - num_test_examples
block_size = int(1.0 * 10**5)

# Generate test set as the last 500,000 rows (same as in paper)
df = pandas.read_csv(input_filename, skiprows=num_train_examples, nrows=num_test_examples)
X_test = df.values[:, 1:]
y_test = numpy.array(df.values[:, 0], numpy.int)

# standardise data
scaler = StandardScaler()
df = pandas.read_csv(input_filename, skiprows=0, nrows=block_size)
X_train = df.values[:, 1:]
y_train = numpy.array(df.values[:, 0], numpy.int)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def largescale(X_train, y_train, X_test, y_test):
    # Feature generation using random forests
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest.fit(X_train, y_train)
    encoder = OneHotEncoder()
    encoder.fit(forest.apply(X_train))
    X_test = encoder.transform(forest.apply(X_test))
    learner = SGDClassifier(loss="log", penalty="l2", learning_rate="constant", l1_ratio=0.1, alpha=10.0**-6, average=10**4, eta0=1.0)

    num_passes = 5
    aucs = []

    for j in range(num_passes):
        for i in range(0, num_train_examples, block_size):
            df = pandas.read_csv(input_filename, skiprows=i, nrows=block_size)
            X_train = df.values[:, 1:]
            X_train = scaler.transform(X_train)
            X_train = encoder.transform(forest.apply(X_train))
            y_train = numpy.array(df.values[:, 0], numpy.int)

            learner.partial_fit(X_train, y_train, classes=numpy.array([0, 1]))
            y_pred_prob = learner.decision_function(X_test)
            auc = roc_auc_score(y_test, y_pred_prob)
            aucs.append([i + block_size * j, auc])
            print(aucs[-1])

    return pandas.DataFrame(aucs)

aucs = largescale(X_train, y_train, X_test, y_test)
print(aucs)
