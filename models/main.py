import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import ClassifierChain

# use this path from Matteo's laptop
data_path = "..\\data\\features_extracted.csv"

data_raw = pd.read_csv(data_path)
data_raw.set_index("Unnamed: 0", inplace=True)

# print(data_raw["label"])
data_raw.drop("label", axis=1, inplace=True)

data_raw.reset_index(drop=True, inplace=True)

label = list(data_raw["label"])
y = [lab.split(" ") for lab in label]
y = MultiLabelBinarizer().fit_transform(y)
print(type(data_raw.values))
classifier = ClassifierChain(svm.SVC())
scores = cross_val_score(classifier, data_raw.values.astype(float), y, cv=10)
print(scores)
