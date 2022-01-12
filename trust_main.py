import pandas as pd
import sklearn as sk
import numpy as np
import time
import os

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
from TrustCalculator import LimeTrust, EntropyTrust, SHAPTrust, NeighborsTrust, \
    ExternalTrust, CombinedTrust, MultiCombinedTrust, ConfidenceInterval

from sklearn import datasets
from keras.datasets import mnist


def process_image_dataset(dataset_name):
    if dataset_name == "MNIST":
        mn = datasets.load_digits(as_frame=True)
        labels = mn.target_names
        x_mnist = mn.frame
        y_mnist = mn.target
        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_mnist, y_mnist, test_size=0.5, shuffle=True)
        return x_mnist, y_mnist, x_tr, x_te, y_tr, y_te, labels

    elif dataset_name == "MNIST-BIG":
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        x_tr = [x.flatten() for x in x_tr]
        x_te = [x.flatten() for x in x_te]
        labels = np.unique(y_tr)
        x_mnist = np.concatenate((x_tr, x_te), axis=0)
        y_mnist = np.concatenate((y_tr, y_te), axis=0)
        return x_mnist, y_mnist, pd.DataFrame(x_tr[0:10000]), pd.DataFrame(x_te), y_tr[0:10000], y_te, labels


def process_tabular_dataset(dataset_name, label_name, limit_rows):
    """
    Method to process an input dataset as CSV
    :param limit_rows: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return:
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Testing Purposes
    if (np.isfinite(limit_rows)) & (limit_rows < len(df.index)):
        df = df[0:limit_rows]

    print("Dataset loaded: " + str(len(df.index)) + " items")
    encoding = pd.factorize(df[label_name])
    y_enc = encoding[0]
    labels = encoding[1]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("Dataset loaded: " + str(len(df.index)) + " items, " + str(len(normal_frame.index)) +
          " normal and " + str(len(labels)) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

    return x_no_cat, y_enc, x_tr, x_te, y_tr, y_te, labels


def load_config(file_config):
    """
    Method to load configuration parameters from input file
    :param file_config: name of the config file
    :return: array with 3 items: [dataset files, label name, max number of rows (if correctly specified, NaN otherwise)]
    """
    config = configparser.RawConfigParser()
    config.read(file_config)
    config_file = dict(config.items('CONFIGURATION'))

    # Processing paths
    path_string = config_file['path']
    if ',' in path_string:
        path_string = path_string.split(',')
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
    return datasets_path, config_file['label'], lim_rows


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


if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Loading Configuration
    dataset_files, y_label, limit_rows = load_config("config.cfg")

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and (dataset_file != "MNIST") and (dataset_file != "MNIST-BIG"):
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")
        else:
            print("Processing Dataset " + dataset_file)
            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                X, y, X_train, X_test, y_train, y_test, label_tags = process_tabular_dataset(dataset_file, y_label,
                                                                                             limit_rows)
            else:
                # Reading Non-Tabular Dataset (@LEONARDO)
                X, y, X_train, X_test, y_train, y_test, label_tags = process_image_dataset(dataset_file)

            xt_numpy = X_test.to_numpy()

            classifiers = [
                # GBClassifier(),
                # DecisionTree(depth=100),
                # KNeighbors(k=11),
                LDA(),
                # LogisticReg(),
                # Bayes(),
                # RandomForest(trees=100),
                # NeuralNetwork(num_input=len(X_test.values[0]), num_classes=len(label_tags))
            ]

            print("Preparing Trust Calculators...")

            # Trust Calculators
            calculators = [
                EntropyTrust(norm=len(label_tags)),
                LimeTrust(X_train.to_numpy(), y_train, X_train.columns, label_tags, 100),
                SHAPTrust(xt_numpy, 100),
                NeighborsTrust(x_train=X_train, y_train=y_train, k=19, labels=label_tags),
                ExternalTrust(del_clf=Bayes(), x_train=X_train, y_train=y_train, norm=len(label_tags)),
                CombinedTrust(del_clf=GBClassifier(), x_train=X_train, y_train=y_train, norm=len(label_tags)),
                MultiCombinedTrust(clf_set=[Bayes(), LDA(), LogisticReg()],
                                   x_train=X_train, y_train=y_train, norm=len(label_tags)),
                MultiCombinedTrust(clf_set=[RandomForest(trees=10), GBClassifier(), DecisionTree(depth=100)],
                                   x_train=X_train, y_train=y_train, norm=len(label_tags)),
                ConfidenceInterval(x_train=X_train.to_numpy(), y_train=y_train, confidence_level=0.9999)
            ]

            for classifierModel in classifiers:
                classifierName = classifierModel.classifier_name()
                print("\n-----------------------------------------------------------------------------------------"
                      "\nProcessing Dataset '" + dataset_file + "' with classifier: " + classifierName + "\n")

                start_ms = current_ms()
                classifierModel.fit(X_train, y_train)
                train_ms = current_ms()
                y_pred = classifierModel.predict_class(X_test)
                test_time = current_ms() - train_ms
                y_proba = classifierModel.predict_prob(X_test)

                # Classifier Evaluation
                print(
                    classifierName + " train/test in " + str(train_ms - start_ms) + "/" + str(
                        test_time) + " ms with Accuracy: "
                    + str(sk.metrics.accuracy_score(y_test, y_pred)))

                # Output Dataframe
                out_df = X_test.copy()
                out_df['true_label'] = list(map(lambda x: label_tags[x], y_test))
                out_df['predicted_label'] = list(map(lambda x: label_tags[x], y_pred))
                out_df['is_misclassification'] = np.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
                out_df['probabilities'] = [y_proba[i] for i in range(len(X_test))]

                for calculator in calculators:
                    print("Calculating Trust Strategy: " + calculator.trust_strategy_name())
                    start_ms = current_ms()
                    trust_scores = calculator.trust_scores(xt_numpy, y_proba, classifierModel)
                    if type(trust_scores) is dict:
                        for key in trust_scores:
                            out_df[calculator.trust_strategy_name() + ' - ' + str(key)] = trust_scores[key]
                    else:
                        out_df[calculator.trust_strategy_name()] = trust_scores
                    print("Completed in " + str(current_ms() - start_ms) + " ms")

                file_out = 'output_folder/' + clean_name(dataset_file) + "_" + classifierName + '.csv'
                print("Printing File '" + file_out + "'")
                out_df.to_csv(file_out, index=False)
                print("Print Completed")
