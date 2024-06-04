import copy
import os

import joblib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import LogisticReg, get_classifier_name, build_classifier, choose_classifier, \
    UnsupervisedClassifier
from sprout.utils.dataset_utils import process_tabular_dataset, process_image_dataset, is_image_dataset, \
    process_binary_tabular_dataset
from sprout.utils.general_utils import load_config, clean_name, current_ms, clear_folder
from sprout.utils.sprout_utils import build_SPROUT_dataset

# Vars for Generating Uncertainties
FILE_AVOID_TAG = None
MODEL_TYPE = 'UNS'
GENERATE_UNCERTAINTIES = False

# Vars for Learning Model
MODELS_FOLDER = "../models/"
STUDY_TAG = {
    "uns_all": "./datasets_measures/unsup/all",
    #"uns_bin": "./datasets_measures/unsup/bindatasets",
    #"uns_multi": "./datasets_measures/unsup/nonbindatasets",
}

MISC_RATIOS = [None, 0.05, 0.1, 0.2, 0.3]


def compute_datasets_uncertainties(dataset_files, d_folder, s_folder,
                                   classifier_list, y_label, limit_rows):
    """
    Computes Uncertainties for many datasets.
    """

    for dataset_file in dataset_files:

        if (dataset_file is None) or (len(dataset_file) == 0) or \
                ((not os.path.isfile(dataset_file)) and not is_image_dataset(dataset_file)):

            # Error while Reading Dataset
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")

        elif (FILE_AVOID_TAG is not None) and (FILE_AVOID_TAG in dataset_file):

            # Dataset to be skipped
            print("Dataset '" + str(dataset_file) + "' skipped due to tag avoidance")

        else:

            print("Processing Dataset " + dataset_file + (
                " - limit " + str(limit_rows) if np.isfinite(limit_rows) else ""))

            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                if MODEL_TYPE == 'SUP':
                    x_train, x_test, y_train, y_test, label_tags, features = \
                        process_tabular_dataset(dataset_file, y_label, limit_rows)
                else:
                    x_train, x_test, y_train, y_test, label_tags, features = \
                        process_binary_tabular_dataset(dataset_file, y_label, limit_rows)
            else:
                # Other / Image Dataset
                x_train, x_test, y_train, y_test, label_tags, features = process_image_dataset(dataset_file, limit_rows)

            print("Preparing Trust Calculators...")
            if MODEL_TYPE == 'SUP':
                contamination = 0
                sp_obj = build_supervised_object(x_train, y_train, label_tags)
            else:
                contamination = sum(y_train) / len(y_train)
                y_train = None
                sp_obj = build_unsupervised_object(x_train, contamination)

            for classifier_string in classifier_list:
                # Building  and  exercising  classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy", contamination)

                sprout_obj = copy.deepcopy(sp_obj)
                sprout_obj.add_calculator_bagging(base_clf=classifier, x_train=x_train, y_train=y_train,
                                                  n_base=5, max_features=0.7, sampling_ratio=0.5,
                                                  perc_decisors=0.5, n_decisors=None,
                                                  n_classes=len(label_tags) if label_tags is not None else 2)
                sprout_obj.add_calculator_boosting(base_clf=classifier, x_train=x_train, y_train=y_train,
                                                   n_base=5, learning_rate=None, sampling_ratio=0.3,
                                                   contamination=None, conf_thr=0.9,
                                                   n_classes=len(label_tags) if label_tags is not None else 2)

                if classifier is not None:
                    y_proba, y_pred = build_classifier(classifier, x_train, y_train, x_test, y_test)

                    # Initializing SPROUT dataset for output
                    out_df = build_SPROUT_dataset(y_proba, y_pred, y_test, label_tags)

                    # Calculating Trust Measures with SPROUT
                    q_df = sprout_obj.compute_set_trust(data_set=x_test, classifier=classifier)
                    out_df = pd.concat([out_df, q_df], axis=1)

                    # Printing Dataframe
                    file_out = s_folder + clean_name(dataset_file, d_folder) + "_" + get_classifier_name(
                        classifier) + '.csv'
                    if not os.path.exists(os.path.dirname(file_out)):
                        os.mkdir(os.path.dirname(file_out))
                    out_df.to_csv(file_out, index=False)
                    print("File '" + file_out + "' Printed")
                else:
                    print("Unable to recognize classifier '" + classifier_string + "'")


def load_uncertainty_datasets(datasets_folder, train_split=0.5, avoid_tags=[], perf_thr=None,
                              label_name="is_misclassification", clean_data=True):
    big_data = []
    for file in os.listdir(datasets_folder):
        if file.endswith(".csv") and (
                (avoid_tags is None) or (len(avoid_tags) == 0) or not any(x in file for x in avoid_tags)):
            df = pandas.read_csv(datasets_folder + "/" + file)
            if perf_thr is not None:
                df_acc = 1 - df[label_name].mean()
                if df_acc >= perf_thr:
                    big_data.append(df)
                else:
                    print("File '" + file + "' discarded: has Accuracy " + str(df_acc) + " < " + str(perf_thr))
            else:
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

    print("Dataset contains " + str(len(label)) + " items and " + str(misc_frac * 100) + "% of misclassifications")

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


def build_supervised_object(x_train, y_train, label_tags):
    sp_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    if (x_train is not None) and isinstance(x_train, pandas.DataFrame):
        x_data = x_train.to_numpy()
    else:
        x_data = x_train
    sp_obj.add_calculator_confidence(x_train=x_data, y_train=y_train, confidence_level=0.9)
    sp_obj.add_calculator_maxprob()
    sp_obj.add_calculator_entropy(n_classes=len(label_tags) if label_tags is not None else 2)
    sp_obj.add_calculator_external(classifier=Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
                                   x_train=x_data, y_train=y_train,
                                   n_classes=len(label_tags) if label_tags is not None else 2)
    sp_obj.add_calculator_combined(classifier=XGBClassifier(n_estimators=30), x_train=x_data, y_train=y_train,
                                   n_classes=len(label_tags) if label_tags is not None else 2)
    for cc in [[Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())]),
                LinearDiscriminantAnalysis(), LogisticReg()],
               [Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())]),
                Pipeline([("norm", MinMaxScaler()), ("clf", BernoulliNB())]),
                Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
                Pipeline([("norm", MinMaxScaler()), ("clf", ComplementNB())])],
               [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=10),
                GradientBoostingClassifier(n_estimators=10)]]:
        sp_obj.add_calculator_multicombined(clf_set=cc, x_train=x_data, y_train=y_train,
                                            n_classes=len(label_tags) if label_tags is not None else 2)
    sp_obj.add_calculator_neighbour(x_train=x_data, y_train=y_train, label_names=label_tags)
    sp_obj.add_calculator_proximity(x_train=x_data, n_iterations=10, range=0.1)
    sp_obj.add_calculator_proximity(x_train=x_data, n_iterations=20, range=0.05)
    sp_obj.add_calculator_recloss(x_train=x_data)

    return sp_obj


def build_unsupervised_object(x_train, contamination):
    contamination = contamination if contamination <= 0.5 else 0.5
    sp_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    if (x_train is not None) and isinstance(x_train, pandas.DataFrame):
        x_data = x_train.to_numpy()
    else:
        x_data = x_train
    sp_obj.add_calculator_confidence(x_train=x_data, y_train=None, confidence_level=0.5)
    sp_obj.add_calculator_maxprob()
    sp_obj.add_calculator_entropy(n_classes=2)
    sp_obj.add_calculator_neighbour(x_train=x_data, y_train=None, label_names=["normal", "anomaly"])
    sp_obj.add_calculator_proximity(x_train=x_data)
    sp_obj.add_calculator_combined(classifier=UnsupervisedClassifier(PCA(contamination=contamination)),
                                   x_train=x_data, y_train=None, n_classes=2)
    for cc in [[UnsupervisedClassifier(COPOD(contamination=contamination)),
                UnsupervisedClassifier(ECOD(contamination=contamination))],
               [UnsupervisedClassifier(CBLOF(contamination=contamination, alpha=0.75, beta=3)),
                UnsupervisedClassifier(PCA(contamination=contamination)),
                UnsupervisedClassifier(MCD(contamination=contamination, support_fraction=0.9)),
                UnsupervisedClassifier(HBOS(contamination=contamination, n_bins=30))],
               [UnsupervisedClassifier(IForest(contamination=contamination)),
                UnsupervisedClassifier(INNE(contamination=contamination))]]:
        sp_obj.add_calculator_multicombined(clf_set=cc, x_train=x_data, y_train=None, n_classes=2)
    sp_obj.add_calculator_recloss(x_train=x_data)
    return sp_obj


if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Reading preferences
    dataset_files, dataset_folder, sprout_folder, sup_clfs, uns_clfs, y_label, limit_rows = load_config("config.cfg")

    # Generating Input data for training Misclassification Predictors
    if not os.path.exists(sprout_folder):
        os.mkdir(sprout_folder)

    if GENERATE_UNCERTAINTIES or len(os.listdir(sprout_folder)) == 0:
        compute_datasets_uncertainties(dataset_files, dataset_folder, sprout_folder,
                                       sup_clfs if MODEL_TYPE == 'SUP' else uns_clfs,
                                       y_label, limit_rows)
    if MODEL_TYPE == 'SUP':
        sprout_obj = build_supervised_object(None, None, None)
    else:
        sprout_obj = build_unsupervised_object(None, 0.1)

    sprout_obj.add_calculator_bagging(base_clf=None, x_train=None, y_train=None)
    sprout_obj.add_calculator_boosting(base_clf=None, x_train=None, y_train=None)

    for tag, folder_path in STUDY_TAG.items():

        print("---------------------------------------------------------\n"
              "           Analysis using tag:" + tag + "\n"
                                                       "---------------------------------------------------------\n")

        if os.path.exists(folder_path):
            # Merging data into a unique Dataset for training Misclassification Predictors
            x_train, y_train, x_test, y_test, features, m_frac = \
                load_uncertainty_datasets(folder_path, train_split=0.75, perf_thr=0.7)

            # Classifiers for Detection (Binary Adjudicator)
            m_frac = 0.5 if m_frac > 0.5 else m_frac
            CLASSIFIERS = [GradientBoostingClassifier(n_estimators=10),
                           GradientBoostingClassifier(n_estimators=20),
                           DecisionTreeClassifier(),
                           LinearDiscriminantAnalysis(),
                           RandomForestClassifier(n_estimators=10),
                           RandomForestClassifier(n_estimators=20),
                           GaussianNB(),
                           LogisticReg(),
                           ]

            # Training Binary Adjudicators to Predict Misclassifications
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
                    start_ms = current_ms()
                    clf.fit(x_tr, y_tr)
                    end_ms = current_ms()
                    y_pred = clf.predict(x_test)
                    mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
                    print("[" + clf_name + "][ratio=" + str(ratio) + "] Accuracy: " + str(
                        sklearn.metrics.accuracy_score(y_test, y_pred))
                          + " and MCC of " + str(mcc) + " in " + str((end_ms - start_ms) / 60000) + " mins")
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

            # Setting up folder to store the SPROUT model
            models_details_folder = MODELS_FOLDER + tag + "/"
            if not os.path.exists(models_details_folder):
                os.mkdir(models_details_folder)
            else:
                clear_folder(models_details_folder)

            # Stores details of the SPROUT object used to build the Binary Adjudicator
            sprout_obj.save_object(models_details_folder)

            # Storing the classifier to be used for Predicting Misclassifications of a Generic Classifier.
            model_file = models_details_folder + "binary_adj_model.joblib"
            joblib.dump(best_clf, model_file, compress=9)

            # Tests if storing was successful
            clf_obj = joblib.load(model_file)
            y_p = clf_obj.predict(x_test)
            if sklearn.metrics.matthews_corrcoef(y_test, y_p) == best_metrics["MCC"]:
                print("Model stored successfully at '" + model_file + "'")
            else:
                print("Error while storing the model - file corrupted")

            # Scores of the SPROUT wrapper
            det_dict = {"analysis tag": tag,
                        "binary classifier": get_classifier_name(best_clf),
                        "train data size": len(y_train),
                        "train data features": numpy.asarray(features),
                        "original misclassification ratio of training set": m_frac,
                        "actual misclassification ratio in training set": best_ratio,
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
            with open(models_details_folder + "binary_adjudicator_metrics.txt", 'w') as f:
                for key, value in det_dict.items():
                    f.write('%s:%s\n' % (key, value))

            # Plot ROC_AUC
            sklearn.metrics.RocCurveDisplay.from_estimator(best_clf, x_test, y_test)
            plt.savefig(models_details_folder + "binary_adjudicator_aucroc_plot.png")

            if hasattr(best_clf, "feature_importances_"):
                f_imp = dict(zip(numpy.asarray(features), best_clf.feature_importances_))
            elif hasattr(best_clf, "coef_"):
                fi_scores = abs(best_clf.coef_[0])
                f_imp = dict(zip(numpy.asarray(features), fi_scores / sum(fi_scores)))
            else:
                print("No feature importance can be computed for " + get_classifier_name(best_clf))
                f_imp = {}
            with open(models_details_folder + "binary_adjudicator_feature_importances.csv", 'w') as f:
                for key, value in f_imp.items():
                    f.write('%s,%s\n' % (key, value))

        else:
            print("Path for the analysis does not exist")
