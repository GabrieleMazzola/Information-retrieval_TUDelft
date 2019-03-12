import pickle
from os import listdir
from os.path import isfile, join

import numpy as np

mypath = "./"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
results = [f for f in onlyfiles if f.endswith(".p")]

res_dicts = []
for file in results:
    with open(f"./{file}", 'rb') as fin:
        res = pickle.load(fin)
        res_dicts.append(res)

# Accuracy
for res_dict in res_dicts:
    for model_configuration_key, scores in res_dict['All']['accuracy']:
        print(model_configuration_key, np.mean(scores), np.std(scores))
