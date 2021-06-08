import numpy as np
ls = [[{"speaker": "1", "phrase_position": "final", "vowel": "a"}, 
{"speaker": "1", "phrase_position": "final", "vowel": "eh"}], [{"speaker": "2", "phrase_position": "final", "vowel": "a"}, 
{"speaker": "2", "phrase_position": "final", "vowel": "eh"}], [{"speaker": "3", "phrase_position": "final", "vowel": "a"}, 
{"speaker": "3", "phrase_position": "final", "vowel": "eh"}]]

print(ls[0][0]["speaker"])

print(ls.pop())

count = 0
for i in range(50):
    count+=1
    if count >45:
        print(count)
    else:
        pass

a = "3.34535"
b = float(a)

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ls_data = [{"F1": 345, "F2": 645, "F3": 1200}, 
{"F1": 200, "F2": 500, "F3": 944}, {"F1": 700, "F2": 900, "F3": 2350}, 
{"F1": 899, "F2": 1487, "F3": 3000}, {"F1": 457, "F2": 800, "F3": 1500}, 
{"F1": 320, "F2": 834, "F3": 1002}, {"F1": 900, "F2": 1834, "F3": 3200}, 
{"F1": 788, "F2": 1700, "F3": 2800}]

value_ls = []
all_value = []
for item in ls_data:
    for value in item.values():
        value_ls.append(value)
    all_value.append(value_ls)
    value_ls = []
#print(all_value)

ls_data_2 = [[ 345,  745, 1300], 
[np.nan, np.nan, np.nan], [700, 900, 2750], 
[899, 1287, np.nan], [457, np.nan, 1500], 
[320, 834, 1002], [900, 1834, 3200], 
[788, 1700, 2800]]

def data_imputation(input_ls):
    imput = IterativeImputer(imputation_order = "random", random_state = 0)
    imputed_features = imput.fit_transform(input_ls)
    return imputed_features

ls_update = []
for i, row in enumerate(ls_data):
    for j, key in enumerate(row):
        row[key] = ls_data_2[i][j]
    ls_update.append(row)
#print(ls_update)

bined_features = KBinsDiscretizer(n_bins=[5, 10, 20], encode='ordinal', strategy='uniform')
ls_bined_data = bined_features.fit_transform(data_imputation(ls_data_2))
#print(data_imputation(ls_data_2))
#print(ls_bined_data)


from typing import Dict

input_diction = {"F1": 345, "F2": 645, "F3": 1200} 
def dic_value_printer(input: Dict):
    new_dict = {}
    new_dict["f1"] = input["F1"]
    return new_dict

print(dic_value_printer(input_diction))

