import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from res.util import get_subset_features


def test_on_all(test_function, filename_dict):
    ir_data = pd.read_csv("../../data/extracted_Features.csv")
    ir_data.drop('Unnamed: 0', inplace=True, axis=1)

    label = list(ir_data["label"])
    y = [lab.split(" ") for lab in label]
    y = MultiLabelBinarizer().fit_transform(y)

    ir_data.drop("label", inplace=True, axis=1)

    cont_feat = ['InitSim', 'DlgSim', 'QuestMark', 'Dup', 'What', 'Where', 'When', 'Why', 'Who', 'How']
    struct_feat = ['AbsPos', 'NormPos', 'Len', 'LenUni', 'LenStem', 'Starter']
    sent_feat = ['Thank', 'ExMark', 'Feedback', 'SenScr(Neg)', 'SenScr(Neu)', 'SenScr(Pos)', 'Lex(Pos)', 'Lex(Neg)']

    feat_to_test = []
    feat_to_test.append(('All', struct_feat + sent_feat + cont_feat))

    types_of_score = ['accuracy', 'precision_samples', 'recall_samples', 'f1_samples']

    res_dict = {}
    for name, feature_names in feat_to_test:
        df = get_subset_features(ir_data, feature_names)
        feat_dict = {}
        print("\n-----------------")
        print(f"Subset shape: {df.shape}")
        for evaluation in types_of_score:
            print(f"Testing {name} features, evaluation: {evaluation}")
            model_names, mod_results = test_function(df, y, evaluation)
            feat_dict[evaluation] = list(zip(model_names, mod_results))
        res_dict[name] = feat_dict
        print("-----------------\n")

    with open(filename_dict, 'wb') as f:
        pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
