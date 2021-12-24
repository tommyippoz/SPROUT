import pandas as pd
import sklearn as sk
import numpy as np

from Classifier import GBClassifier
from Classifier import DecisionTree
from Classifier import KNeighbors
from Classifier import LDA
from Classifier import LogisticReg
from Classifier import Bayes
from Classifier import RandomForest
from Classifier import CSupportVector

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
    df['predicted_label'] = y_pred
    df['probabilities'] = [y_proba[i] for i in range(len(X_test))]
    for calculator in calculators:
        trust_scores = calculator.trust_scores(xt_numpy, y_proba)
        df[calculator.trust_strategy_name()] = trust_scores

    df.to_csv('output_folder/' + classifierName + '.csv', index=False)



if __name__ == '__main__':
    dataset_file, y_label = load_config("config.cfg")
    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(dataset_file, y_label)
    # Trust Calculators
    calculators = [
        EntropyTrust()
    ]
    # Building Classifier
    classifiers = [
        GBClassifier(X_train, y_train, X_test),
        DecisionTree(X_train, y_train, X_test),
        KNeighbors(X_train, y_train, X_test),
        LDA(X_train, y_train, X_test),
        LogisticReg(X_train, y_train, X_test),
        Bayes(X_train, y_train, X_test),
        RandomForest(X_train, y_train, X_test),
        CSupportVector(X_train, y_train, X_test)
    ]
    # Output Dataframe
    for classifier in classifiers:
        print(type(classifier))
        classifierName = classifier.classifier_name()
        print(classifierName)
        y_pred = classifier.predict_class()
        print(y_pred)
        y_proba = classifier.predict_prob()
        print(y_proba)
        # Classifier Evaluation
        print(classifierName + " Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))
        # print_to_csv(X_test, y_pred, y_proba, calculators, classifierName)


    # explainer = LimeTrust(X_train.to_numpy(), y_train, X.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(xt_numpy, y_proba))
