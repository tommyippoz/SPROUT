import copy
import os

import joblib
import numpy
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from sprout.utils.Classifier import FastAI, LogisticReg, XGB, TabNet
from sprout.utils.dataset_utils import process_tabular_dataset, process_image_dataset, is_image_dataset
from sprout.utils.general_utils import load_config, choose_classifier, clean_name
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.sprout_utils import build_classifier, build_SPROUT_dataset, get_classifier_name

import matplotlib.pyplot as plt

# Vars for Generating Uncertainties
INTERMEDIATE_FOLDER = "./datasets_measures/NIDS/"
GENERATE_UNCERTAINTIES = False
FILE_AVOID_TAG = None

# Vars for Learning Model
ANALYSIS_AVOID_TAGS = {"all": None, "no_dt": ["DecisionTree"], "no_dt_lr": ["DecisionTree", "Logistic"]}
MODELS_FOLDER = "../models/"
MODEL_NAME = "nids"
MISC_RATIOS = [None, 0.05, 0.1, 0.2]


def compute_datasets_uncertainties(dataset_files, classifier_list, y_label, limit_rows, out_folder):
    """
    Computes Uncertainties for many datasets.
    """

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not is_image_dataset(dataset_file):

            # Error while Reading Dataset
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")

        elif (FILE_AVOID_TAG is not None) and (FILE_AVOID_TAG in dataset_file):

            # Dataset to be skipped
            print("Dataset '" + str(dataset_file) + "' skipped due to tag avoidance")

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
                                                     [GaussianNB(), BernoulliNB(),
                                                      Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
                                                      Pipeline([("norm", MinMaxScaler()), ("clf", ComplementNB())])],
                                                     [DecisionTreeClassifier(), RandomForestClassifier(), XGB()],
                                                     #[FastAI(), TabNetClassifier()]
                                                     ])

            for classifier_string in classifier_list:

                # Building and exercising classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                y_proba, y_pred = build_classifier(classifier, x_train, y_train, x_test, y_test)

                # Initializing SPROUT dataset for output
                out_df = build_SPROUT_dataset(y_proba, y_pred, y_test, label_tags)

                # Calculating Trust Measures with SPROUT
                q_df = quail.compute_set_trust(data_set=x_test, classifier=classifier)
                out_df = pd.concat([out_df, q_df], axis=1)

                # Printing Dataframe
                file_out = out_folder + '/' + clean_name(dataset_file) + "_" + get_classifier_name(classifier) + '.csv'
                out_df.to_csv(file_out, index=False)
                print("File '" + file_out + "' Printed")


def load_uncertainty_datasets(datasets_folder, train_split=0.5, avoid_tags=[],
                              label_name="is_misclassification", clean_data=True):
    big_data = []
    for file in os.listdir(datasets_folder):
        if file.endswith(".csv") and ((avoid_tags is None) or (len(avoid_tags) == 0) or not any(x in file for x in avoid_tags)):
            df = pandas.read_csv(datasets_folder + "/" + file)
            big_data.append(df)
    big_data = pandas.concat(big_data)

    big_data = big_data.sample(frac=1.0)
    big_data = big_data.fillna(0)
    big_data = big_data.replace('null', 0)

    # Cleaning Data
    if clean_data:
        big_data = big_data.drop(columns=["true_label", "predicted_label"])
        big_data = big_data.select_dtypes(exclude=['object'])

    # Creating train-test split
    label = big_data[label_name].to_numpy()
    misc_frac = sum(label) / len(label)
    big_data = big_data.drop(columns=[label_name])
    features = big_data.columns
    big_data = big_data.to_numpy()
    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(big_data, label, train_size=train_split)

    print("Dataset contains " + str(len(label)) + " items and " + str(misc_frac*100) + "% of misclassifications")

    return x_tr, y_tr, x_te, y_te, features, misc_frac


def sample_data(x, y, ratio):

    # Creating DataFrame
    df = pd.DataFrame(x.copy())
    df["is_misclassification"] = y
    normal_frame = df.loc[df["is_misclassification"] == 0]
    misc_frame = df.loc[df["is_misclassification"] == 1]

    # Scaling to 'ratio'
    df_ratio = len(misc_frame.index) / len(normal_frame.index)
    if df_ratio < ratio:
        normal_frame = normal_frame.sample(frac=(df_ratio / (2 * ratio)))
    df = pd.concat([normal_frame, misc_frame])
    df = df.sample(frac=1.0)

    return df.drop(["is_misclassification"], axis=1).to_numpy(), df["is_misclassification"].to_numpy()


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
        compute_datasets_uncertainties(dataset_files, classifier_list, y_label, limit_rows, INTERMEDIATE_FOLDER)

    for analysis_desc in ANALYSIS_AVOID_TAGS:

        print("---------------------------------------------------------\n"
              "Analysis using tag:" + analysis_desc + "\n"
              "---------------------------------------------------------\n")

        # Merging data into a unique Dataset for training Misclassification Predictors
        x_train, y_train, x_test, y_test, features, m_frac = \
            load_uncertainty_datasets(INTERMEDIATE_FOLDER,
                                      avoid_tags=ANALYSIS_AVOID_TAGS[analysis_desc],
                                      train_split=0.5)

        # Classifiers for Detection
        m_frac = 0.5 if m_frac > 0.5 else m_frac
        CLASSIFIERS = [XGB(),
                       #RandomForestClassifier(),
                       #COPOD(contamination=m_frac),
                       #CBLOF(contamination=m_frac),
                       #TabNet(metric="auc", verbose=0),
                       #FastAI(feature_names=features, label_name="is_misclassification", verbose=2, metric="roc_auc")
                       ]

        # Training Binary Classifiers to Predict Misclassifications
        best_clf = None
        best_ratio = None
        best_metrics = {"MCC": -10}

        # Formatting MISC_RATIOS
        if MISC_RATIOS is None or len(MISC_RATIOS) == 0:
            MISC_RATIOS = [None]

        for clf_base in CLASSIFIERS:
            clf_name = get_classifier_name(clf_base)
            for ratio in MISC_RATIOS:
                clf = copy.deepcopy(clf_base)
                if ratio is not None:
                    x_tr, y_tr = sample_data(x_train, y_train, ratio)
                else:
                    x_tr = x_train
                    y_tr = y_train
                clf.fit(x_tr, y_tr)
                y_pred = clf.predict(x_test)
                mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
                print("[" + clf_name + "][ratio=" + str(ratio) + "] Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " and MCC of " + str(mcc))
                if mcc > best_metrics["MCC"]:
                    best_ratio = ratio
                    best_clf = clf
                    [tn, fp], [fn, tp] = sklearn.metrics.confusion_matrix(y_test, y_pred)
                    best_metrics = {"MCC": mcc,
                                    "Accuracy": sklearn.metrics.accuracy_score(y_test, y_pred),
                                    "AUC ROC": sklearn.metrics.roc_auc_score(y_test, y_pred),
                                    "Precision": sklearn.metrics.precision_score(y_test, y_pred),
                                    "Recall": sklearn.metrics.recall_score(y_test, y_pred),
                                    "TP": tp,
                                    "TN": tn,
                                    "FP": fp,
                                    "FN": fn}

        print("\nBest classifier is " + get_classifier_name(best_clf) + "/" + str(best_ratio) +
              " with MCC = " + str(best_metrics["MCC"]))

        # Storing the classifier to be used for Predicting Misclassifications of a Generic Classifier.
        model_file = MODELS_FOLDER + MODEL_NAME + "_" + analysis_desc + ".joblib"
        joblib.dump(best_clf, model_file)

        # Tests if storing was successful
        clf_obj = joblib.load(model_file)
        y_p = clf_obj.predict(x_test)
        if sklearn.metrics.matthews_corrcoef(y_test, y_p) == best_metrics["MCC"]:
            print("Model stored successfully at '" + model_file + "'")
        else:
            print("Error while storing the model - file corrupted")

        # Printing files with Details
        det_dict = {"analysis tag": analysis_desc,
                    "classifier": get_classifier_name(best_clf),
                    "train data size": len(y_train),
                    "train data features": numpy.asarray(features),
                    "original misclassification ratio": m_frac,
                    "actual misclassification ratio": best_ratio,
                    "test_mcc": best_metrics["MCC"],
                    "test_acc": best_metrics["Accuracy"],
                    "test_auc": best_metrics["AUC ROC"],
                    "test_p": best_metrics["Precision"],
                    "test_r": best_metrics["Recall"],
                    "test_tp": best_metrics["TP"],
                    "test_tn": best_metrics["TN"],
                    "test_fp": best_metrics["FP"],
                    "test_fn": best_metrics["FN"],
                    }
        with open(MODELS_FOLDER + MODEL_NAME + "_" + analysis_desc + ".txt", 'w') as f:
            for key, value in det_dict.items():
                f.write('%s:%s\n' % (key, value))

        # Plot ROC_AUC
        sklearn.metrics.plot_roc_curve(best_clf, x_test, y_test)
        plt.savefig(MODELS_FOLDER + MODEL_NAME + "_" + analysis_desc + "_aucroc.png")

        f_imp = dict(zip(numpy.asarray(features), best_clf.feature_importances_))
        with open(MODELS_FOLDER + MODEL_NAME + "_" + analysis_desc + "_feature_importances.csv", 'w') as f:
            for key, value in f_imp.items():
                f.write('%s,%s\n' % (key, value))
