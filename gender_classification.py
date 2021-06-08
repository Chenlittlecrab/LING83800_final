#!/usr/bin/env python
"""Gender classification using vowel and fricative data"""

import csv
import logging
import statistics
import glob
import numpy as np

from typing import Dict, List, Tuple

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
    imput = IterativeImputer(imputation_order = "random", random_state = 0)
    imputed_features = imput.fit_transform(input_ls)
    return imputed_features

def preprocess_features(feature_ls: List[Dict]):
    value_ls: List[float] = []
    all_value: List[value_ls] = []
    for item in feature_ls:
        for value in item.values():
            value_ls.append(value)
        all_value.append(value_ls)
        value_ls = []
    if len(all_value[0]) == 3: #number of features for vowels
        discretizer = KBinsDiscretizer(n_bins=[15, 30, 15], encode='ordinal', strategy='quantile')
    if len(all_value[0]) == 4: #number of features for fricatives
        discretizer = KBinsDiscretizer(n_bins=[30, 150, 60, 175], encode='ordinal', strategy='quantile')
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
    correct: List[int] = []
    size: List[int] = []
    logging.basicConfig(format="%(levelname)s: %(message)s", level = "INFO")
    for train_path in glob.iglob(TRAIN_TXT):
        vectorizer = sklearn.feature_extraction.DictVectorizer()
        #training
        (feature_vectors, y) = extract_features_file(train_path)
        x = vectorizer.fit_transform(feature_vectors)
        model = sklearn.linear_model.LogisticRegression(
            penalty = "l1",
            C = 10,
            solver = "liblinear",
        )
        model.fit(x,y)
        test_path = train_path.replace("train", "test")
        #evaluation
        (feature_vectors, y) = extract_features_file(test_path)
        x = vectorizer.transform(feature_vectors)
        y_predict = model.predict(x)
        assert len(y) == len(y_predict), "Lengths don't match"
        correct.append(sum(y == y_predict))
        size.append(len(y))
    #accuracy info
    micro_accuracy = sum(correct) / sum(size)
    print(micro_accuracy)


if __name__ == "__main__":
    main()

    



