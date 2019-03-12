import itertools as it

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.adapt import MLkNN


def test_features(df, truth, eval_type):
    models = [('LDA', LinearDiscriminantAnalysis())]

    res = []
    names = []
    for mod_name, model in models:
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(ClassifierChain(model), df.values, truth, cv=kfold, scoring=eval_type)
        res.append(cv_results)
        names.append(mod_name)
        msg = "%s: %f (%f)" % (mod_name, cv_results.mean(), cv_results.std())
        print(msg)
    return names, res


def get_subset_features(df, feat_names):
    return df[feat_names].copy()


def test_AdaBoost(df, truth, eval_type):
    param_adaBoost = {'n_estimators': [50, 100]}
    combinations_adaBoost = it.product(*(param_adaBoost[Name] for Name in param_adaBoost))

    # Test the combinations for AdaBoost with cross validation
    results_adaBoost = []
    keys = []
    for index, values in enumerate(combinations_adaBoost):
        key = "ADA" + "-".join([str(item) for item in values])

        clf = AdaBoostClassifier(n_estimators=values[0])
        classifier = ClassifierChain(clf)
        kfold = KFold(n_splits=10, random_state=26)
        scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)
        keys.append(key)
        results_adaBoost.append(scores)

        msg = "%s: %f (%f)" % (key, scores.mean(), scores.std())
        print(msg)

    return keys, results_adaBoost


def test_randForest(df, truth, eval_type):
    param_randForest = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 50, 80, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']}
    combinations_randForest = it.product(*(param_randForest[Name] for Name in param_randForest))

    # Test the combinations for Random Forest with cross validation
    results_randForest = []
    keys = []
    for index, values in enumerate(combinations_randForest):
        key = "RF" + "-".join([str(item) for item in values])

        clf = RandomForestClassifier(n_estimators=values[0], max_depth=values[1], max_features=values[2],
                                     criterion=values[3])
        classifier = ClassifierChain(clf)
        kfold = KFold(n_splits=10, random_state=26)
        scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)
        keys.append(key)
        results_randForest.append(scores)

        msg = "%s: %f (%f)" % (key, scores.mean(), scores.std())
        print(msg)

    return keys, results_randForest


def test_best_rf(df, truth, eval_type):
    clf = RandomForestClassifier(n_estimators=200, max_depth=50, max_features='auto',
                                 criterion='entropy')
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    print("Start crossvalidation...")
    scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)
    print(f"Crossvalidation done. Mean: {np.mean(scores)}")
    return scores


def test_best_AdaBoost(df, truth, eval_type):
    clf = AdaBoostClassifier(n_estimators=50)
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    print("Start crossvalidation...")
    scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)
    print(f"Crossvalidation done. Mean: {np.mean(scores)}")
    return scores


def test_svm(df, truth, eval_type):
    # param_svm = {'C': [1, 10, 100, 1000],
    #              'kernel': ['linear', 'rbf'],
    #              'gamma': ['auto', 'scale']
    #              }
    param_svm = {'C': [100],
                 'kernel': ['linear'],
                 'gamma': ['auto']
                 }
    combinations_svm = it.product(*(param_svm[Name] for Name in param_svm))

    # Test the combinations for SVM with cross validation
    results_svm = []
    keys = []
    for index, values in enumerate(combinations_svm):
        key = "SVM" + "-".join([str(item) for item in values])

        clf = svm.SVC(C=values[0], kernel=values[1], gamma=values[2])
        classifier = ClassifierChain(clf)
        kfold = KFold(n_splits=10, random_state=26)
        scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)
        keys.append(key)
        results_svm.append(scores)

        msg = "%s: %f (%f)" % (key, scores.mean(), scores.std())
        print(msg)

    return keys, results_svm


def test_naiveBayes(df, truth, eval_type):
    clf = MultinomialNB()
    classifier = ClassifierChain(clf)
    kfold = KFold(n_splits=10, random_state=26)
    scores = cross_val_score(classifier, df.values, truth, cv=kfold, scoring=eval_type)

    return ["NB"], [scores]


# def test_mlknn(df, truth, eval_type):
#     parameters = {'k': range(1, 4), 's': [0.5, 0.7, 1.0]}
#     kfold = KFold(n_splits=10, random_state=26)
#     print("Start gridsearch")
#     clf = GridSearchCV(MLkNN(), parameters, scoring=eval_type, cv=kfold)
#     clf.fit(df, truth)
#     print(f"Gridsearch completed. Best params: {clf.best_params_}")
#
#
#     best_classifier = MLkNN(clf.best_params_)
#     print("Start Crossval")
#     scores = cross_val_score(best_classifier, df.values, truth, cv=kfold, scoring=eval_type)
#     return ["MLkNN"], [scores]

def test_mlknn(df, truth, eval_type):
    parameters = {'k': range(1, 4), 's': [0.5, 0.7, 1.0]}
    kfold = KFold(n_splits=10, random_state=26)
    # print("Start gridsearch")
    # clf = GridSearchCV(MLkNN(), parameters, scoring=eval_type, cv=kfold)
    # clf.fit(df, truth)
    # print(f"Gridsearch completed. Best params: {clf.best_params_}")

    best_classifier = MLkNN(k=3, s=0.5)
    print("Start Crossval")
    scores = cross_val_score(best_classifier, df.values, truth, cv=kfold, scoring=eval_type)
    return ["MLkNN"], [scores]
