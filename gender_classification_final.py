#!/usr/bin/env python
"""Gender classification using vowel and fricative data"""

import csv
import logging
import glob
import numpy as np

from typing import Dict, List, Tuple

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.dummy import DummyClassifier
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.svm



FeatureVector = Dict[str, float]
FeatureVectors = List[FeatureVector]

TRAIN_TXT = "data/*_train.txt"

def _segment_features(feature_string: str):
    if feature_string == "missing":
        return np.nan
    else:
        return float(feature_string)


def extract_features(path: str, input_dict: Dict):
    """Extracts feature vector for each speaker"""
    features: Dict[str, str] = {}
    if path.startswith("data/vowel"):
        features["F1"] = _segment_features(input_dict["F1_bark"])
        features["F2"] = _segment_features(input_dict["F2_bark"])
        features["F3"] = _segment_features(input_dict["F3_bark"])
    if path.startswith("data/fricative"):
        features["center_of_gravity"] = _segment_features(input_dict["cog"])
        features["standard_deviation"] = _segment_features(input_dict["sdev"])
        features["skewness"] = _segment_features(input_dict["skew"])
        features["kurtosis"] = _segment_features(input_dict["kurt"])
    return features    

def data_imputation(input_ls: List[List]):
    """Imputes missing values using multiple imputation method."""
    imput = IterativeImputer(imputation_order = "random", random_state = 0)
    imputed_features = imput.fit_transform(input_ls)
    return imputed_features

def preprocess_features(feature_ls: List[Dict]):
    """This function discretizes continuous variables into ordinals"""
    value_ls: List[float] = []
    all_value: List[value_ls] = []
    #extracts all the values from the dictionaries stored in the list
    #and put values of each dict in a list, and append those lists
    #into a whole list. 
    for item in feature_ls:
        for value in item.values():
            value_ls.append(value)
        all_value.append(value_ls)
        value_ls = []
    #discretize the values using KbinsDiscretizer in sklean
    if len(all_value[0]) == 3: #number of features for vowels
        discretizer = KBinsDiscretizer(n_bins=[25, 25, 25], encode='ordinal', strategy='uniform')
    if len(all_value[0]) == 4: #number of features for fricatives
        discretizer = KBinsDiscretizer(n_bins=[175, 150, 60, 30], encode='ordinal', strategy='uniform')
    binned_feature_ls = discretizer.fit_transform(
        data_imputation(all_value)
        ) #call data_imputation function to impute the missing data
    
    #put the converted values back to the dictionary
    feature_ls_binned: List[Dict] = []
    for i, item in enumerate(feature_ls):
        for j, key in enumerate(item):
            item[key] = binned_feature_ls[i][j]
        feature_ls_binned.append(item)
    return feature_ls_binned

def extract_features_file(input_path: str) -> Tuple[FeatureVectors, List[str]]:
    """Extracts feature vectors for both vowel and fricative tsv files."""
    features: FeatureVectors = []
    labels: List[str] = []
    with open(input_path, "r") as source:
        for row in csv.DictReader(source, delimiter = "\t"):
            labels.append(row["sex"])
            features.append(extract_features(input_path, row))
    return preprocess_features(features), labels

def main() -> None:
    correct_linear: List[int] = []
    correct_svm: List[int] = []
    correct_dummy: List[int] = []
    size: List[int] = []
    logging.basicConfig(format="%(levelname)s: %(message)s", level = "INFO")
    for train_path in glob.iglob(TRAIN_TXT):
        vectorizer = sklearn.feature_extraction.DictVectorizer()
        #training
        (feature_vectors, y) = extract_features_file(train_path)
        x = vectorizer.fit_transform(feature_vectors)
        model_linear = sklearn.linear_model.LogisticRegression(
            penalty = "l1",
            C = 10,
            solver = "liblinear",
        )
        model_svm = sklearn.svm.SVC(
            kernel = "poly",
            degree = 3,
            max_iter=-1, #prevent the liblinear not converge issue
            tol=0.001
        )
        model_dummy = DummyClassifier(
            strategy = "uniform"
        )
        model_linear.fit(x,y)
        model_svm.fit(x,y)
        model_dummy.fit(x, y)
        test_path = train_path.replace("train", "test")
        #evaluation
        (feature_vectors, y) = extract_features_file(test_path)
        x = vectorizer.transform(feature_vectors)
        y_linear_predict = model_linear.predict(x)
        y_svm_predict = model_svm.predict(x)
        y_dummy = model_dummy.predict(x)
        assert len(y) == len(y_linear_predict), "Lengths don't match"
        assert len(y) == len(y_svm_predict), "Lenghs don't match"
        correct_linear.append(sum(y == y_linear_predict))
        correct_svm.append(sum(y == y_svm_predict))
        correct_dummy.append(sum(y == y_dummy))
        size.append(len(y))
    #accuracy info
    logging.info("Linear regression method accuracy:\t%.4f", sum(correct_linear) / sum(size))
    logging.info("SVC method accuracy:\t%.4f", sum(correct_svm) / sum(size))
    logging.info("Dummy method accuracy:\t%.4f", sum(correct_dummy) / sum(size))


if __name__ == "__main__":
    main()

    



