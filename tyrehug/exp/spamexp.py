#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import pandas
import sklearn
import sklearn.feature_extraction
import sklearn.svm

df = pandas.read_csv("../data/spam/SMSSpamCollection", sep="\t", header=-1)

tf_transformer = sklearn.feature_extraction.text.TfidfVectorizer()
X = tf_transformer.fit_transform(df[1])
y = pandas.get_dummies(df[0], drop_first=True).values.ravel()

learner = sklearn.svm.LinearSVC()
scores = sklearn.cross_validation.cross_val_score(learner, X, y, cv=5, scoring="accuracy")
print(scores.mean())
scores = sklearn.cross_validation.cross_val_score(learner, X, y, cv=5, scoring="recall")
print(scores.mean())
scores = sklearn.cross_validation.cross_val_score(learner, X, y, cv=5, scoring="precision")
print(scores.mean())
