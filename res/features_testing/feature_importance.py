import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



ir_data = pd.read_csv("../../data/processed_dataset.csv")
ir_data.drop('Unnamed: 0', inplace=True, axis=1)

label = list(ir_data["label"])
y_lab = [lab.split(" ") for lab in label]
bin = MultiLabelBinarizer()
y = bin.fit_transform(y_lab)

ir_data.drop("label", inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(ir_data, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=200, max_depth=50, max_features='auto',
                                 criterion='entropy')
#classifier = LabelPowerset(clf)
clf.fit(X_train, y_train)
#imp = classifier.feature_importances_
#print(imp)


feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
#predictions = classifier.predict(X_test)





