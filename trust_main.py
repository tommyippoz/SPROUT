import pandas as pd
import sklearn as sk
import numpy as np

from xgboost import XGBClassifier

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


if __name__ == '__main__':

    dataset_file, y_label = load_config("config.cfg")

    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(dataset_file, y_label)

    # Building Classifier
    classifierName = "XGBoost"
    classifierModel = XGBClassifier()
    classifierModel.fit(X_train, y_train)
    y_pred = classifierModel.predict(X_test)
    y_proba = classifierModel.predict_proba(X_test)

    # Classifier Evaluation
    print(classifierName + " Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))

    # Trust Calculators
    calculators = [
        EntropyTrust()
    ]

    # Output Dataframe
    xt_numpy = X_test.to_numpy()
    df = X_test
    df['true_label'] = y_test
    df['predicted_label'] = y_pred
    df['probabilities'] = [y_proba[i] for i in range(len(y_test))]

    for calculator in calculators:
        trust_scores = calculator.trust_scores(xt_numpy, y_proba)
        df[calculator.trust_strategy_name()] = trust_scores

    df.to_csv('output_folder/out_frame.csv', index=False)

    # explainer = LimeTrust(X_train.to_numpy(), y_train, X.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(xt_numpy, y_proba))
