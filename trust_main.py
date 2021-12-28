import pandas as pd
import sklearn as sk
import numpy as np
import time

from Classifier import GBClassifier
from Classifier import DecisionTree
from Classifier import KNeighbors
from Classifier import LDA
from Classifier import LogisticReg
from Classifier import Bayes
from Classifier import RandomForest
from Classifier import SupportVectorMachine
from Classifier import NeuralNetwork

import configparser
from TrustCalculator import LimeTrust, NativeTrust, EntropyTrust, SHAPTrust


def process_dataset(dataset_name, label_name):
    """
    Method to process an input dataset as CSV
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return:
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Testing Purposes
    df = df[0:1000]

    print("Dataset loaded: " + str(len(df.index)) + " items")
    y_multi = df[label_name]
    y_bin = np.where(y_multi == "normal", 0, 1)

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("Normal data points: " + str(len(normal_frame.index)) + " items ")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_bin, test_size=0.5, shuffle=True)

    return x_no_cat, y_bin, x_tr, x_te, y_tr, y_te


def load_config(file_config):
    config = configparser.RawConfigParser()
    config.read(file_config)
    config_file = dict(config.items('CONFIGURATION'))
    return config_file['path'], config_file['label']


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


'''

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


def otherMain():
    dataset_file, y_label = load_config("config.cfg")
    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(dataset_file, y_label)
    # Trust Calculators
    calculators = [
        EntropyTrust()
    ]
    # Building Classifiers
    classifiers = [
        # GBClassifier(X_train, y_train, X_test),
        # DecisionTree(X_train, y_train, X_test),
        # KNeighbors(X_train, y_train, X_test),
        # LDA(X_train, y_train, X_test),
        # LogisticReg(X_train, y_train, X_test),
        # Bayes(X_train, y_train, X_test),
        # RandomForest(X_train, y_train, X_test),
        # CSupportVector(X_train, y_train, X_test),
        NeuralNetwork(X_train, y_train, X_test)
    ]
    # Output Dataframe
    for classifier in classifiers:
        classifierName = classifier.classifier_name()
        y_pred = classifier.predict_class()
        y_proba = classifier.predict_prob()
        print(y_proba)
        # Classifier Evaluation
        print(classifierName + " Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))
        # Write CSV
        write2csv(X_test, calculators, classifierName, y_test, y_pred, y_proba)
        # print_to_csv(X_test, y_pred, y_proba, calculators, classifierName)


    # explainer = LimeTrust(X_train.to_numpy(), y_train, X.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(xt_numpy, y_proba))

'''

if __name__ == '__main__':

    # Loading Configuration
    dataset_file, y_label = load_config("config.cfg")

    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(dataset_file, y_label)
    xt_numpy = X_test.to_numpy()

    classifiers = [
        # GBClassifier(),
        # DecisionTree(depth=100),
        # KNeighbors(k=10),
        # LDA(),
        # LogisticReg(),
        # Bayes(),
        # RandomForest(trees=100),
        # SupportVectorMachine(kernel='linear', degree=1),
        NeuralNetwork(X_train, y_train, X_test)
    ]

    # Trust Calculators
    calculators = [
        EntropyTrust(),
        NativeTrust(),
        LimeTrust(X_train.to_numpy(), y_train, X_train.columns, ['normal', 'attack'], 100),
        SHAPTrust(xt_numpy, 100)
    ]

    for classifierModel in classifiers:
        classifierName = classifierModel.classifier_name()
        print("Processing Dataset with classifier: " + classifierName)

        start_ms = current_ms()
        classifierModel.fit(X_train, y_train)
        train_ms = current_ms()
        y_pred = classifierModel.predict_class(X_test)
        test_time = current_ms() - train_ms
        y_proba = classifierModel.predict_prob(X_test)

        # Classifier Evaluation
        print(
            classifierName + " train/test in " + str(train_ms - start_ms) + "/" + str(test_time) + " ms with Accuracy: "
            + str(sk.metrics.accuracy_score(y_test, y_pred)))

        # Output Dataframe
        out_df = X_test.copy()
        out_df['true_label'] = np.where(y_test == 0, "normal", "anomaly")
        out_df['predicted_label'] = np.where(y_pred == 0, "normal", "anomaly")
        out_df['is_FP'] = np.where((out_df['true_label'] == 'normal') & (out_df['predicted_label'] == 'anomaly'), 1, 0)
        out_df['is_FN'] = np.where((out_df['true_label'] == 'anomaly') & (out_df['predicted_label'] == 'normal'), 1, 0)
        out_df['is_misclassification'] = out_df['is_FP'] + out_df['is_FN']
        out_df['probabilities'] = [y_proba[i] for i in range(len(X_test))]

        for calculator in calculators:
            print("Calculating Trust Strategy: " + calculator.trust_strategy_name())
            trust_scores = calculator.trust_scores(xt_numpy, y_proba, classifierModel)
            if type(trust_scores) is dict:
                for key in trust_scores:
                    out_df[calculator.trust_strategy_name() + ' - ' + str(key)] = trust_scores[key]
            else:
                out_df[calculator.trust_strategy_name()] = trust_scores

        file_out = 'output_folder/' + classifierName + '_new.csv'
        print("Printing File '" + file_out + "'")
        out_df.to_csv(file_out, index=False)
        print("Print Completed")
