import sklearn
import csv
import pandas as pd
import numpy as np
from skmultilearn.problem_transform import ClassifierChain
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer

# use this path from Matteo's laptop
data_path = "..\\data\\processed_dataset.csv"

data_raw = pd.read_csv(data_path)
data_raw.set_index("Unnamed: 0", inplace=True)

#print(data_raw["label"])
data_raw.drop("label", axis=1, inplace=True)

data_raw.reset_index(drop=True, inplace=True)

label = list(data_raw["label"])
y = [lab.split(" ") for lab in label]
y = MultiLabelBinarizer().fit_transform(y)
print(type(data_raw.values))
classifier = ClassifierChain(svm.SVC())
scores = cross_val_score(classifier, data_raw.values.astype(float), y, cv=10)
print(scores)