import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain




ir_data = pd.read_csv("../../data/processed_dataset.csv")
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

diffr_y_test = []
diffr_predicted_labels = []

print(y_test[0] == predicted_labels[0])

for i in range(0, len(y_test)):
    if not list_comparison(y_test[i], predicted_labels[i]):
        diffr_y_test.append(y_test[i])
        diffr_predicted_labels.append(predicted_labels[i])


d = {"True labels": list(diffr_y_test), "Predicted labels": list(diffr_predicted_labels)}

analysis = pd.DataFrame(d)
print(analysis)

analysis.to_csv("partial_misclassification_analysis.csv")

print(f"Total length: {len(y_test)}")
print(f"Misclassified length: {len(diffr_y_test)}")



