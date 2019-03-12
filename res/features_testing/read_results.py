import pickle
from os import listdir
from os.path import isfile, join
import numpy as np


with open("./features_test.p", 'rb') as fin:
    res = pickle.load(fin)


for feat_type in res.keys():
    for eval_type in res[feat_type].keys():
        scores = res[feat_type][eval_type]
        print(f"{feat_type} -> {eval_type} -> {np.mean(scores)}")

    print("\n")
