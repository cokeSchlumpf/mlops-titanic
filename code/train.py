import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Result Dictionary
results = {}

# Split into x and y
train = pd.read_pickle('../data/train_prepared.pkl')
x_test = pd.read_pickle('../data/test_prepared.pkl')
x_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs', max_iter = 1000)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
results['Logistic Regression (ACC)'] = acc_log

# Support Vector Machines
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
results['Support Vector Machines (ACC)'] = acc_svc

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
results['Decission Tree (ACC)'] = acc_decision_tree

with open('../data/metrics.json', 'w') as fp:
    json.dump(results, fp, indent=2, sort_keys=True)