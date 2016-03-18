import pandas
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

input_filename = "../data/spam/SMSSpamCollection"
input_file = open(input_filename)

labels = []
messages = []

for line in input_file.readlines():
    words = line.split("\t")
    labels.append(words[0])
    messages.append(words[1])

tf_transformer = TfidfVectorizer()
X = tf_transformer.fit_transform(messages)
y = pandas.get_dummies(labels, drop_first=True).values.ravel()

learner = LinearSVC()
scores = cross_val_score(learner, X, y, cv=5, scoring="accuracy")
print(scores.mean())
scores = cross_val_score(learner, X, y, cv=5, scoring="recall")
print(scores.mean())
scores = cross_val_score(learner, X, y, cv=5, scoring="precision")
print(scores.mean())
