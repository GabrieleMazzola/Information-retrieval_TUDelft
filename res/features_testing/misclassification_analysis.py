import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

ir_data = pd.read_csv("../../data/extracted_Features.csv")
ir_data.drop('Unnamed: 0', inplace=True, axis=1)

label = list(ir_data["label"])
y_lab = [lab.split(" ") for lab in label]
bin = MultiLabelBinarizer()
y = bin.fit_transform(y_lab)

ir_data.drop("label", inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(ir_data, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def list_comparison(list1, list2):
    for ind in range(0, len(list1)):
        if list1[ind] != list2[ind]:
            return False

    return True


clf = AdaBoostClassifier(n_estimators=50)
classifier = ClassifierChain(clf)

classifier.fit(X_train, y_train)
predicted_labels = classifier.predict(X_test)

