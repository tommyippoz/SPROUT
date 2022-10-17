import configparser
import os
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sprout.utils.Classifier import XGB, TabNet, KNeighbors, LogisticReg, FastAI


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
    elif clf_name in {"DT", "DTree", "DecisionTree"}:
        return DecisionTreeClassifier()
    elif clf_name in {"KNN", "knn", "kNN", "KNeighbours"}:
        return KNeighbors(k=11)
    elif clf_name in {"SVM"}:
        return BaggingClassifier(SVC(gamma='auto', probability=True), max_samples=0.1, n_estimators=10)
    elif clf_name in {"LDA"}:
        return LinearDiscriminantAnalysis()
    elif clf_name in {"Regression", "LogisticRegression", "LR"}:
        return LogisticReg()
    elif clf_name in {"RF", "RandomForest"}:
        return RandomForestClassifier(n_estimators=10)
    elif clf_name in {"TabNet", "Tabnet", "TN"}:
        return TabNet(metric="auc", verbose=2)
    elif clf_name in {"FAI", "FastAI", "FASTAI", "fastai"}:
        return FastAI(feature_names=features, label_name=y_label, metric=metric)
    elif clf_name in {"GBC", "GradientBoosting"}:
         return GradientBoostingClassifier(n_estimators=50)
    else:
        pass


def get_full_class_name(class_obj):
    return class_obj.__module__ + "." + class_obj.__qualname__
