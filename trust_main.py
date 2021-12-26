import pandas as pd
import sklearn as sk
import numpy as np

import csv
from csv import writer

from Classifier import GBClassifier
from Classifier import DecisionTree
from Classifier import KNeighbors
from Classifier import LDA
from Classifier import LogisticReg
from Classifier import Bayes
from Classifier import RandomForest
from Classifier import CSupportVector
from Classifier import NeuralNetwork

from TrustCalculator import LimeTrust
from TrustCalculator import EntropyTrust
import configparser


def process_dataset(dataset_name, y_label):
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")
    print("Dataset loaded: " + str(len(df.index)) + " items")
    y = df[y_label]
    y_bin = np.where(y == "normal", "normal", "attack")
    # Basic Pre-Processing
    attack_labels = df[y_label].unique()
    normal_frame = df.loc[df[y_label] == "normal"]
    print("Normal data points: " + str(len(normal_frame.index)) + " items ")
    x = df.drop(columns=[y_label])
    x_no_cat = x.select_dtypes(exclude=['object'])
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_bin, test_size=0.5, shuffle=True)
    # Training/Testing Classifiers
    return x_no_cat, y_bin, x_tr, x_te, y_tr, y_te


def load_config(file_config):
    config = configparser.RawConfigParser()
    config.read(file_config)
    config_file = dict(config.items('CONFIGURATION'))
    return config_file['path'], config_file['label']


def print_to_csv(X_test, y_pred, y_proba, calculators, classifierName):
    xt_numpy = X_test.to_numpy()
    df = X_test
    df['true_label'] = y_test
    df['predicted_label'] = y_pred
    df['probabilities'] = [y_proba[i] for i in range(len(X_test))]
    for calculator in calculators:
        trust_scores = calculator.trust_scores(xt_numpy, y_proba)
        print(trust_scores)
        df[calculator.trust_strategy_name()] = trust_scores
    df.to_csv('output_folder/' + classifierName + '.csv', index=False)


def write2csv(X_test, calculators, classifierName, y_test, y_pred, y_proba):
    array_trust = cal_trust(calculators, X_test, y_proba)
    file_out = 'output_folder/' + classifierName + '_new.csv'
    array_col = list(X_test.columns)
    create_column(file_out, array_col, calculators)
    array_zeros = np.zeros(len(X_test))
    array_print = np.column_stack((X_test.values, y_test, y_pred, array_zeros, np.asarray(array_trust).T))
    array_print[:, len(X_test.values[0]) + 2] = list(map(str, y_proba))
    dump2csv(file_out, array_print)


def cal_trust(calculators, X_test, y_proba):
    xt_numpy = X_test.to_numpy()
    array_trust = [[0. for x in range(len(X_test))] for y in range(len(calculators))]
    for i in range(len(calculators)):
        array_trust[i][:] = calculators[i].trust_scores(xt_numpy, y_proba)
    return array_trust


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def dump2csv(file_out, matrix):
    with open(file_out, 'a') as csvfile:
        matrixwriter = csv.writer(csvfile, delimiter=',')
        for row in matrix:
            matrixwriter.writerow(row)


def create_column(file_out, labels, calculators):
    f = open(file_out, "w+")
    test = ["true_label", "predict_label", "probabilities"]
    calculator_array = [calculator.trust_strategy_name() for calculator in calculators]
    label = labels + test + calculator_array
    csv.DictWriter(f, fieldnames=label).writeheader()
    f.close()


if __name__ == '__main__':
    dataset_file, y_label = load_config("config.cfg")
    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(dataset_file, y_label)
    # Trust Calculators
    calculators = [
        EntropyTrust()
    ]
    # Building Classifiers
    classifiers = [
        GBClassifier(X_train, y_train, X_test),
        DecisionTree(X_train, y_train, X_test),
        KNeighbors(X_train, y_train, X_test),
        LDA(X_train, y_train, X_test),
        LogisticReg(X_train, y_train, X_test),
        Bayes(X_train, y_train, X_test),
        RandomForest(X_train, y_train, X_test),
        # CSupportVector(X_train, y_train, X_test),
        # NeuralNetwork(X_train, y_train, X_test)
    ]
    # Output Dataframe
    for classifier in classifiers:
        classifierName = classifier.classifier_name()
        y_pred = classifier.predict_class()
        print(y_pred)
        y_proba = classifier.predict_prob()
        print(y_proba)
        # Classifier Evaluation
        print(classifierName + " Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))
        # Write CSV
        write2csv(X_test, calculators, classifierName, y_test, y_pred, y_proba)
        # print_to_csv(X_test, y_pred, y_proba, calculators, classifierName)


    # explainer = LimeTrust(X_train.to_numpy(), y_train, X.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(xt_numpy, y_proba))
