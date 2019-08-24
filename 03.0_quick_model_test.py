# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train_processed.csv")

y_train = pd.Series.from_csv('y_train.csv')

X_test = pd.read_csv("X_test_processed.csv")

y_test = pd.Series.from_csv('y_test.csv')
# -

X_train = X_train.drop('Unnamed: 0', axis=1)
X_test = X_test.drop('Unnamed: 0', axis=1)

y_train.value_counts()

# +
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'run_time'])

models = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=10), SGDClassifier()]
names = ["Naive Bayes", "Decision Tree", "Random Forest Classifier", "SGD Classifier"]


for model, name in zip(models, names):
    temp_list = []
    print(name)
    start = time.time()
    for score in ["accuracy", "precision", "recall"]:
        mean_score = cross_val_score(model, X_train, y_train,scoring=score, cv=5).mean()
        print("{} mean : {}".format(score, mean_score))
        temp_list.append(mean_score)
    temp_list.append(time.time() - start)
    print("time to run: {}".format(time.time() - start))
    print("\n")
    scores_df.loc[name] = temp_list
# -

scores_df

for i in ['accuracy', 'precision', 'recall', 'run_time']:
    scores_df.plot.bar(y = i) 


