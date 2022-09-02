import os

import joblib
import numpy as np
import pandas
import pandas as pd
import sklearn
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier

from sprout.utils.Classifier import FastAI, LogisticReg, XGB
from sprout.utils.dataset_utils import process_tabular_dataset, process_image_dataset, is_image_dataset
from sprout.utils.general_utils import load_config, choose_classifier, clean_name
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.sprout_utils import build_classifier, build_QUAIL_dataset, get_classifier_name

INTERMEDIATE_FOLDER = "./output_folder"
MODELS_FOLDER = "../models/"
GENERATE_UNCERTAINTIES = False


def compute_datasets_uncertainties(dataset_files, classifier_list, y_label, limit_rows, out_folder):
    """
    Computes Uncertainties for many datasets.
    """

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not is_image_dataset(dataset_file):

            # Error while Reading Dataset
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")

        else:

            print("Processing Dataset " + dataset_file + (" - limit " + str(limit_rows) if np.isfinite(limit_rows) else ""))

            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                x_train, x_test, y_train, y_test, label_tags, features = \
                    process_tabular_dataset(dataset_file, y_label, limit_rows)
            else:
                # Other / Image Dataset
                x_train, x_test, y_train, y_test, label_tags, features = process_image_dataset(dataset_file, limit_rows)

            print("Preparing Trust Calculators...")
            quail = SPROUTObject()
            quail.add_all_calculators(x_train=x_train,
                                      y_train=y_train,
                                      label_names=label_tags,
                                      combined_clf=XGB(),
                                      combined_clfs=[[GaussianNB(), LinearDiscriminantAnalysis(), LogisticReg()],
                                                     [GaussianNB(), BernoulliNB(), MultinomialNB(), ComplementNB()],
                                                     [DecisionTreeClassifier(), RandomForestClassifier(), XGB()],
                                                     #[FastAI(), TabNetClassifier()]
                                                     ])

            for classifier_string in classifier_list:

                # Building and exercising classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                y_proba, y_pred = build_classifier(classifier, x_train, y_train, x_test, y_test)

                # Initializing SPROUT dataset for output
                out_df = build_QUAIL_dataset(y_proba, y_pred, y_test, label_tags)

                # Calculating Trust Measures with SPROUT
                q_df = quail.compute_set_trust(data_set=x_test, classifier=classifier)
                out_df = pd.concat([out_df, q_df], axis=1)

                # Printing Dataframe
                file_out = out_folder + '/' + clean_name(dataset_file) + "_" + get_classifier_name(classifier) + '.csv'
                out_df.to_csv(file_out, index=False)
                print("File '" + file_out + "' Printed")


def load_uncertainty_datasets(datasets_folder, train_split=0.5, label_name="is_misclassification", clean_data=True):
    big_data = []
    for file in os.listdir(datasets_folder):
        if file.endswith(".csv"):
            df = pandas.read_csv(datasets_folder + "/" + file)
            big_data.append(df)
    big_data = pandas.concat(big_data)

    # Cleaning Data
    if clean_data:
        big_data = big_data.select_dtypes(exclude=['object'])

    # Creating train-test split
    label = big_data[label_name].to_numpy()
    misc_frac = sum(label) / len(label)
    big_data = big_data.drop(columns=[label_name])
    features = big_data.columns
    big_data = big_data.to_numpy()
    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(big_data, label, train_size=train_split)

    return x_tr, y_tr, x_te, y_te, features, misc_frac


if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Reading preferences
    dataset_files, classifier_list, y_label, limit_rows = load_config("config.cfg")

    # Generating Input data for training Misclassification Predictors
    if not os.path.exists(INTERMEDIATE_FOLDER):
        os.mkdir(INTERMEDIATE_FOLDER)
    if GENERATE_UNCERTAINTIES or len(os.listdir(INTERMEDIATE_FOLDER)) == 0:
        compute_datasets_uncertainties(dataset_files, classifier_list, y_label, 2000, INTERMEDIATE_FOLDER)

    # Merging data into a unique Dataset for training Misclassification Predictors
    x_train, y_train, x_test, y_test, features, m_frac = \
        load_uncertainty_datasets(INTERMEDIATE_FOLDER, train_split=0.5)

    # Classifiers for Detection
    m_frac = 0.5 if m_frac > 0.5 else m_frac
    CLASSIFIERS = [XGB(),
                   RandomForestClassifier(),
                   COPOD(),
                   COPOD(contamination=m_frac),
                   CBLOF(),
                   CBLOF(contamination=m_frac),
                   TabNetClassifier(verbose=0),
                   FastAI(feature_names=features, label_name="is_misclassification", verbose=0, metric="accuracy")]

    # Training Binary Classifiers to Predict Misclassifications
    best_clf = None
    best_mcc = -1
    for clf in CLASSIFIERS:
        clf_name = get_classifier_name(clf)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
        print("[" + clf_name + "] Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
              + " and MCC of " + str(mcc))
        if mcc > best_mcc:
            best_clf = clf
            best_mcc = mcc

    print("\nBest classifier is " + get_classifier_name(best_clf) + " with MCC = " + str(best_mcc))

    # Storing the classifier to be used for Predicting Misclassifications of a Generic Classifier.
    model_file = MODELS_FOLDER + "big_clf.joblib"
    joblib.dump(best_clf, model_file)

    # Tests if storing was successful
    clf_obj = joblib.load(model_file)
    y_p = clf_obj.predict(x_test)
    if sklearn.metrics.matthews_corrcoef(y_test, y_p) == best_mcc:
        print("Model stored successfully at '" + model_file + "'")
    else:
        print("Error while storing the model - file corrupted")
    # TBA
