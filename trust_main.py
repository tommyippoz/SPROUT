import pandas as pd
import sklearn as sk
import numpy as np
import os
import utils

from Classifier import XGB, TabNet, FastAI, GBM, MXNet, ADABoostClassifier
from Classifier import DecisionTree
from Classifier import KNeighbors
from Classifier import LDA
from Classifier import LogisticReg
from Classifier import Bayes
from Classifier import RandomForest
from Classifier import SupportVectorMachine

from TrustCalculator import LimeTrust, EntropyTrust, SHAPTrust, NeighborsTrust, \
    ExternalTrust, CombinedTrust, MultiCombinedTrust, ConfidenceInterval

from sklearn import datasets
from sklearn.datasets import fetch_openml


def process_image_dataset(dataset_name, limit):
    if dataset_name == "DIGITS":
        mn = datasets.load_digits(as_frame=True)
        feature_list = mn.columns
        labels = mn.target_names
        x_mnist = mn.frame
        y_mnist = mn.target
        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_mnist, y_mnist, test_size=0.5, shuffle=True)
        return x_mnist, y_mnist, x_tr, x_te, y_tr, y_te, labels, feature_list

    elif dataset_name == "MNIST":
        mnist = fetch_openml('mnist_784')
        y_mnist = np.asarray(list(map(int, mnist.target)), dtype=int)
        x_mnist = np.stack([x.flatten() for x in mnist.data])
        if (np.isfinite(limit)) & (limit < len(x_mnist)):
            x_mnist = x_mnist[0:limit]
            y_mnist = y_mnist[0:limit]
        feature_list = np.arange(0, len(x_mnist[0]), 1)
        labels = pd.Index(np.unique(mnist.target), dtype=object)
        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(pd.DataFrame(x_mnist), y_mnist, test_size=0.1429, shuffle=True)
        return x_mnist, y_mnist, x_tr, x_te, y_tr, y_te, labels, feature_list

    elif dataset_name == "FASHION-MNIST":
        (x_tr, y_tr), (x_te, y_te) = [] # fashion_mnist.load_data()
        x_tr = [x.flatten() for x in x_tr]
        x_te = [x.flatten() for x in x_te]
        np.concatenate(x_tr, axis=0)
        if (np.isfinite(limit)) & (limit < len(x_tr)):
            x_tr = x_tr[0:limit]
            y_tr = y_tr[0:limit]
        feature_list = np.arange(0, len(x_tr[0]), 1)
        labels = np.unique(y_tr)
        x_mnist = np.concatenate((x_tr, x_te), axis=0)
        y_mnist = np.concatenate((y_tr, y_te), axis=0)
        return x_mnist, y_mnist, pd.DataFrame(x_tr), pd.DataFrame(x_te), y_tr, y_te, labels, feature_list


def process_tabular_dataset(dataset_name, label_name, limit):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return:
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

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
    feature_list = x_no_cat.columns
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

    return x_no_cat, y_enc, x_tr, x_te, y_tr, y_te, labels, feature_list


def is_image_dataset(dataset_name):
    return (dataset_name == "DIGITS") or (dataset_name != "MNIST") or (dataset_name != "FASHION-MNIST")


if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Loading Configuration
    dataset_files, y_label, limit_rows = utils.load_config("config.cfg")

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not is_image_dataset(dataset_file):
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")
        else:
            print("Processing Dataset " + dataset_file)
            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                X, y, X_train, X_test, y_train, y_test, label_tags, features = process_tabular_dataset(dataset_file,
                                                                                                       y_label,
                                                                                                       limit_rows)
            else:
                X, y, X_train, X_test, y_train, y_test, label_tags, features = process_image_dataset(dataset_file,
                                                                                                     limit_rows)

            if not isinstance(X_test, np.ndarray):
                xt_numpy = X_test.to_numpy()
            else:
                xt_numpy = X_test

            classifiers = [
                # XGB(),
                # DecisionTree(depth=100),
                # KNeighbors(k=11),
                # LDA(),
                # LogisticReg(),
                # Bayes(),
                # RandomForest(trees=10),
                # NeuralNetwork(num_input=len(X_test.values[0]), num_classes=len(label_tags))
                # TabNet(),
                FastAI(feature_names=features, label_name=y_label),
                # GBM(feature_names=features, label_name=y_label),
                # MXNet(feature_names=features, label_name=y_label)
            ]

            print("Preparing Trust Calculators...")

            # Trust Calculators
            calculators = [
                EntropyTrust(norm=len(label_tags)),
                LimeTrust(X_train, y_train, features, label_tags, 20),
                SHAPTrust(xt_numpy, max_samples=100, items=10, reg="bic"),
                NeighborsTrust(x_train=X_train, y_train=y_train, k=19, labels=label_tags),
                ExternalTrust(del_clf=Bayes(), x_train=X_train, y_train=y_train, norm=len(label_tags)),
                CombinedTrust(del_clf=XGB(), x_train=X_train, y_train=y_train, norm=len(label_tags)),
                MultiCombinedTrust(clf_set=[Bayes(), LDA(), LogisticReg()],
                                   x_train=X_train, y_train=y_train, norm=len(label_tags)),
                MultiCombinedTrust(clf_set=[RandomForest(trees=10), XGB(), DecisionTree(depth=100),
                                            ADABoostClassifier(n_trees=100)],
                                   x_train=X_train, y_train=y_train, norm=len(label_tags)),
                MultiCombinedTrust(clf_set=[LDA(), XGB(), KNeighbors(k=11)],
                                   x_train=X_train, y_train=y_train, norm=len(label_tags)),
                ConfidenceInterval(x_train=X_train.to_numpy(), y_train=y_train, confidence_level=0.9999)
            ]

            for classifierModel in classifiers:
                classifierName = classifierModel.classifier_name()
                print("\n-----------------------------------------------------------------------------------------"
                      "\nProcessing Dataset '" + dataset_file + "' with classifier: " + classifierName + "\n")

                start_ms = utils.current_ms()
                classifierModel.fit(X_train, y_train)
                train_ms = utils.current_ms()
                y_pred = classifierModel.predict_class(xt_numpy)
                test_time = utils.current_ms() - train_ms
                y_proba = classifierModel.predict_prob(xt_numpy)
                if isinstance(y_proba, pd.DataFrame):
                    y_proba = y_proba.to_numpy()

                # Classifier Evaluation
                print(
                    classifierName + " train/test in " + str(train_ms - start_ms) + "/" + str(
                        test_time) + " ms with Accuracy: "
                    + str(sk.metrics.accuracy_score(y_test, y_pred)))

                # Output Dataframe
                out_df = pd.DataFrame()
                out_df['true_label'] = list(map(lambda x: label_tags[x], y_test))
                out_df['predicted_label'] = list(map(lambda x: label_tags[x], y_pred))
                out_df['is_misclassification'] = np.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
                out_df['probabilities'] = [y_proba[i] for i in range(len(X_test))]

                for calculator in calculators:
                    print("Calculating Trust Strategy: " + calculator.trust_strategy_name())
                    start_ms = utils.current_ms()
                    trust_scores = calculator.trust_scores(xt_numpy, y_proba, classifierModel)
                    if type(trust_scores) is dict:
                        for key in trust_scores:
                            out_df[calculator.trust_strategy_name() + "_" + str(utils.current_ms() - start_ms)
                                   + ' - ' + str(key)] = trust_scores[key]
                    else:
                        out_df[calculator.trust_strategy_name()] = trust_scores
                    print("Completed in " + str(utils.current_ms() - start_ms) + " ms")

                file_out = 'output_folder/' + utils.clean_name(dataset_file) + "_" + classifierName + '.csv'
                print("Printing File '" + file_out + "'")
                out_df.to_csv(file_out, index=False)
                print("Print Completed")
