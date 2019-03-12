import itertools as it

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

ir_data = pd.read_csv("../data/extracted_Features.csv")
ir_data.drop('Unnamed: 0', inplace=True, axis=1)

label = list(ir_data["label"])
y = [lab.split(" ") for lab in label]
y = MultiLabelBinarizer().fit_transform(y)

ir_data.drop("label", inplace=True, axis=1)

# Define the possible values for the hyperparameters
param_svm = {'C': [1, 10, 100, 1000],
             'kernel': ['linear', 'rbf'],
             'gamma': ['auto', 'scale']
             }

param_adaBoost = {'n_estimators': [50, 100]}

param_randForest = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 50, 80, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

# Define the hyperparameter combinations to try for each of the classifiers
combinations_svm = it.product(*(param_svm[Name] for Name in param_svm))
combinations_adaBoost = it.product(*(param_adaBoost[Name] for Name in param_adaBoost))
combinations_randForest = it.product(*(param_randForest[Name] for Name in param_randForest))

# Test the combinations for SVM with cross validation
results_svm = []
keys = []
for index, values in enumerate(combinations_svm):
    print(values)
    clf = svm.SVC(C=values[0], kernel=values[1], gamma=values[2])
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    scores = cross_val_score(classifier, ir_data.values, y, cv=kfold, scoring="accuracy")
    keys.append("SVM" + "-".join(values))
    results_svm.append(scores)

# Test the combinations for Random Forest with cross validation
results_randForest = []
keys = []
for index, values in enumerate(combinations_randForest):
    clf = RandomForestClassifier(n_estimators=values[0], max_depth=values[1], max_features=values[2],
                                 criterion=values[3])
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    scores = cross_val_score(classifier, ir_data.values, y, cv=kfold, scoring="accuracy")
    keys.append("RF" + "-".join(values))
    results_randForest.append(scores)

# Test the combinations for AdaBoost with cross validation
results_adaBoost = []
keys = []
for index, values in enumerate(combinations_adaBoost):
    clf = AdaBoostClassifier(n_estimators=values[0])
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    scores = cross_val_score(classifier, ir_data.values, y, cv=kfold, scoring="accuracy")
    keys.append("ADA" + "-".join(values))
    results_adaBoost.append(scores)

# Test MultinomialNB with cross validation
clf = MultinomialNB()
classifier = ClassifierChain(clf)
kfold = KFold(n_splits=10, random_state=26)
scores = cross_val_score(classifier, ir_data.values, y, cv=kfold, scoring="accuracy")

# scores = cross_val_score(classifier, ir_data.values, y, cv=10)
# print(scores)
