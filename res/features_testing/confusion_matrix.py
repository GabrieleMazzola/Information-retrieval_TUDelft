import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
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

clf = AdaBoostClassifier(n_estimators=50)
classifier = ClassifierChain(clf)

model = classifier.fit(X=X_train, Y=y_train)
predictions = classifier.predict(X=X_test)

cm = confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=predictions.argmax(axis=1))

print(cm)
print(bin.classes_)

print(predictions.argmax(axis=1))
