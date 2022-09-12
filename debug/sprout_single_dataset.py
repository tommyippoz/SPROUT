import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from sprout.utils import dataset_utils, sprout_utils
from sprout.utils.Classifier import XGB, LogisticReg
from sprout.utils.general_utils import load_config, choose_classifier, clean_name
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.sprout_utils import get_classifier_name

MODELS_FOLDER = "../models/"
MODEL_TAG = "big_clf_all"
OUTPUT_FOLDER = "./output_folder/"

if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Loading Configuration
    dataset_files, classifier_list, y_label, limit_rows = load_config("config.cfg")

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not dataset_utils.is_image_dataset(dataset_file):

            # Error while Reading Dataset
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")

        else:

            print("Processing Dataset " + dataset_file + (
                " - limit " + str(limit_rows) if np.isfinite(limit_rows) else ""))

            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_tabular_dataset(dataset_file, y_label, limit_rows)
            else:
                # Other / Image Dataset
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_image_dataset(dataset_file, limit_rows)

            print("Preparing Trust Calculators...")
            sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
            sprout_obj.add_all_calculators(x_train=x_train,
                                           y_train=y_train,
                                           label_names=label_tags,
                                           combined_clf=XGB(),
                                           combined_clfs=[[GaussianNB(), LinearDiscriminantAnalysis(), LogisticReg()],
                                                          [GaussianNB(), BernoulliNB(),
                                                           Pipeline(
                                                               [("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
                                                           Pipeline(
                                                               [("norm", MinMaxScaler()), ("clf", ComplementNB())])],
                                                          [DecisionTreeClassifier(), RandomForestClassifier(), XGB()],
                                                          # [FastAI(), TabNetClassifier()]
                                                          ])

            for classifier_string in classifier_list:
                # Building and exercising classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                y_proba, y_pred = sprout_utils.build_classifier(classifier, x_train, y_train, x_test, y_test)

                # Initializing SPROUT dataset for output
                out_df = sprout_utils.build_QUAIL_dataset(y_proba, y_pred, y_test, label_tags)

                # Calculating Trust Measures with SPROUT
                sp_df = sprout_obj.compute_set_trust(data_set=x_test, classifier=classifier)
                sp_df = sp_df.select_dtypes(exclude=['object'])
                out_df = pd.concat([out_df, sp_df], axis=1)

                # Printing Dataframe
                file_out = OUTPUT_FOLDER + clean_name(dataset_file) + "_" + \
                           sprout_utils.get_classifier_name(classifier) + '.csv'
                out_df.to_csv(file_out, index=False)
                print("File '" + file_out + "' Printed")

                predictions_df, clf = sprout_obj.predict_misclassifications(MODEL_TAG, sp_df)

                if "true" in predictions_df.columns:
                    y_pred = predictions_df["pred"]
                    y_true = predictions_df["true"]
                    [tn, fp], [fn, tp] = sklearn.metrics.confusion_matrix(y_true, y_pred)
                    best_metrics = {"MCC": sklearn.metrics.matthews_corrcoef(y_true, y_pred),
                                    "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
                                    "AUC ROC": sklearn.metrics.roc_auc_score(y_true, y_pred),
                                    "Precision": sklearn.metrics.precision_score(y_true, y_pred),
                                    "Recall": sklearn.metrics.recall_score(y_true, y_pred),
                                    "TP": tp,
                                    "TN": tn,
                                    "FP": fp,
                                    "FN": fn}

                print("\nClassifier [" + MODEL_TAG + "]: " + get_classifier_name(clf) +
                      " has MCC = " + str(best_metrics["MCC"]))

            else:

                print("No true label to compute binary classification metrics")
