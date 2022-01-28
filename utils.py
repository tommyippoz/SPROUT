import configparser
import os
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from Classifier import XGB, TabNet, FastAI, GBM, MXNet
from Classifier import KNeighbors
from Classifier import LogisticReg
from Classifier import Bayes


def load_config(file_config):
    """
    Method to load configuration parameters from input file
    :param file_config: name of the config file
    :return: array with 4 items: [dataset files,
                                    classifiers,
                                    label name,
                                    max number of rows (if correctly specified, NaN otherwise)]
    """
    config = configparser.RawConfigParser()
    if os.path.isfile(file_config):
        config.read(file_config)
        config_file = dict(config.items('CONFIGURATION'))
    
        # Processing classifiers
        classifiers = config_file['classifiers']
        if ',' in classifiers:
            classifiers = [x.strip() for x in classifiers.split(',')]
        else:
            classifiers = [classifiers]
    
        # Processing paths
        path_string = config_file['datasets']
        if ',' in path_string:
            path_string = [x.strip() for x in path_string.split(',')]
        else:
            path_string = [path_string]
        datasets_path = []
        for file_string in path_string:
            if os.path.isdir(file_string):
                datasets_path.extend([os.path.join(file_string, f) for f in os.listdir(file_string) if
                                      os.path.isfile(os.path.join(file_string, f))])
            else:
                datasets_path.append(file_string)
    
        # Processing limit to rows
        lim_rows = config_file['limit_rows']
        if not lim_rows.isdigit():
            lim_rows = np.nan
        else:
            lim_rows = int(lim_rows)
        return datasets_path, classifiers, config_file['label_tabular'], lim_rows
    
    else:
        # Config File does not exist
        return ["DIGITS"], ["RF"], "", "no"


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def clean_name(file):
    """
    Method to get clean name of a file
    :param file: the original file path
    :return: the filename with no path and extension
    """
    name = os.path.basename(file)
    if '.' in name:
        name = name.split('.')[0]
    return name



def choose_classifier(clf_name, features, y_label, metric):
    if clf_name in {"XGB", "XGBoost"}:
        return XGB()
    elif clf_name in {"DTree", "DecisionTree"}:
        return DecisionTreeClassifier(depth=100)
    elif clf_name in {"KNN", "knn", "kNN", "KNeighbours"}:
        return KNeighbors(k=11)
    elif clf_name in {"LDA"}:
        return LinearDiscriminantAnalysis()
    elif clf_name in {"NaiveBayes", "Bayes"}:
        return Bayes()
    elif clf_name in {"Regression", "LogisticRegression", "LR"}:
        return LogisticReg()
    elif clf_name in {"RF", "RandomForest"}:
        return RandomForestClassifier(trees=10)
    elif clf_name in {"TabNet", "Tabnet"}:
        return TabNet(metric)
    elif clf_name in {"FastAI", "FASTAI", "fastai"}:
        return FastAI(feature_names=features, label_name=y_label, metric=metric)
    elif clf_name in {"GBM", "LightGBM"}:
        return GBM(feature_names=features, label_name=y_label, metric=metric)
    else:
        pass
