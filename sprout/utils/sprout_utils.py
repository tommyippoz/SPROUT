import copy
import csv
import os.path

import numpy
import numpy as np
import pandas
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.general_utils import current_ms


def build_SPROUT_dataset(x_test, y_proba, y_pred, y_test, label_tags) -> pandas.DataFrame:
    """
    Prepares DataFrame to output SPROUT results
    :param y_proba: probabilities assigned by the classifier
    :param y_pred: predictions (classes) of the classifier
    :param y_test: labels of the test set (ground truth)
    :param label_tags: Names of the classes
    :return: a DataFrame with 4 columns
    """
    out_df = x_test.copy()
    out_df.reset_index(drop=True, inplace=True)
    out_df['true_label'] = list(map(lambda x: label_tags[x], y_test))
    out_df['predicted_label'] = list(map(lambda x: label_tags[x], y_pred))
    out_df['is_misclassification'] = np.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
    out_df['probabilities'] = [np.array2string(y_proba[i], separator=";") for i in range(len(y_proba))]
    return out_df


def read_adjudicator_calculators(model_folder: str) -> dict:
    """
    Reads Uncertainty measures associated to a specific adjudicator to load
    :param model_folder: folder where the adjudicator was saved
    :return: a dictionary
    """
    uc_dict = {}
    if os.path.exists(model_folder + "uncertainty_calculator_params.csv"):
        with open(model_folder + "uncertainty_calculator_params.csv") as csvfile:
            my_reader = csv.reader(csvfile, delimiter=',')
            next(my_reader, None)
            for row in my_reader:
                if row[0] not in uc_dict:
                    uc_dict[row[0].strip()] = {}
                uc_dict[row[0].strip()][row[1].strip()] = row[2].strip()
    return uc_dict


def predictions_variability(predictions: numpy.ndarray):
    """
    Computes the variability of predictions
    :param predictions: numpy 2d matrix of (n_predictions, n_classes)
    :return: variability/entropy of predictions
    """
    vp = 0.0
    if predictions is not None:
        stds = numpy.std(predictions, axis=0)
        vp = sum(stds)
    return vp


def compute_omission_metrics(y_true: numpy.ndarray, y_wrapper: numpy.ndarray, y_clf: numpy.ndarray, reject_tag=None) -> dict:
    """
    Assumes that y_clf may have omissions, labeled as 'reject_tag'
    :param y_true: the ground truth labels
    :param y_wrapper: the prediction of the SPROUT (wrapper) classifier
    :param y_clf: the prediction of the regular classifier
    :param reject_tag: the tag used to label rejections, default is None
    :return: a dictionary of metrics
    """
    met_dict = {}
    met_dict['alpha'] = sklearn.metrics.accuracy_score(y_true, y_clf)
    met_dict['eps'] = 1 - met_dict['alpha']
    met_dict['phi'] = numpy.count_nonzero(y_wrapper == reject_tag) / len(y_true)
    met_dict['alpha_w'] = sum(y_true == y_wrapper) / len(y_true)
    met_dict['eps_w'] = 1 - met_dict['alpha_w'] - met_dict['phi']
    met_dict['phi_c'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf == y_true), 1, 0)) / len(y_true)
    met_dict['phi_m'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf != y_true), 1, 0)) / len(y_true)
    met_dict['eps_gain'] = 0 if met_dict['eps'] == 0 else (met_dict['eps'] - met_dict['eps_w']) / met_dict['eps']
    met_dict['phi_m_ratio'] = 0 if met_dict['phi'] == 0 else met_dict['phi_m'] / met_dict['phi']
    met_dict['overall'] = 2 * met_dict['eps_gain'] * met_dict['phi_m_ratio'] / (
            met_dict['eps_gain'] + met_dict['phi_m_ratio'])
    return met_dict


# Classifiers for Detection (Binary Adjudicator). They are all supervised
CLASSIFIERS = [XGBClassifier(n_estimators=30),
               XGBClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=30),
               GradientBoostingClassifier(n_estimators=100),
               DecisionTreeClassifier(),
               LinearDiscriminantAnalysis(),
               RandomForestClassifier(n_estimators=30),
               RandomForestClassifier(n_estimators=100),
               GaussianNB(),
               LogisticRegression()]


def train_binary_adjudicator(x_train: numpy.ndarray, y_train: numpy.ndarray,
                             x_test: numpy.ndarray, y_test: numpy.ndarray, features: list = None,
                             misc_ratios: list = [None, 0.05, 0.1, 0.2, 0.3], verbose=True):
    """
    Trains a binary adjudicator using uncertainty measures as data, and 0/1 misclassification as label
    :param x_train: uncertainty measures of the train set
    :param y_train: 0/1 misclassification labels of the train set
    :param x_test: uncertainty measures of the validation/test set
    :param y_test: 0/1 misclassification labels of the validation/test set
    :param features: name of the features of data. Used to buld feature importance
    :param misc_ratios: list of rations for up/down-scaling misclassifications when training the adjudicator
    :param verbose: True if debug information has to be shown
    :return: the adjudicator, the feature importance, the metrics on validation/test set
    """

    # Training Binary Adjudicators to Predict Misclassifications
    best_clf = None
    best_ratio = None
    best_metrics = {"MCC": -10}

    # Formatting MISC_RATIOS
    if misc_ratios is None or len(misc_ratios) == 0:
        misc_ratios = [None]
    # Formatting features
    if features is None or len(features) <= 0:
        features = ["feature_" + str(i) for i in range(0, x_train.shape[1])]

    for clf_base in CLASSIFIERS:
        clf_name = get_classifier_name(clf_base)
        for ratio in misc_ratios:
            clf = copy.deepcopy(clf_base)
            if ratio is not None:
                x_tr, y_tr = sample_adj_data(x_train, y_train, ratio)
            else:
                x_tr = x_train
                y_tr = y_train
            start_ms = current_ms()
            clf.fit(x_tr, y_tr)
            end_ms = current_ms()
            y_pred = clf.predict(x_test)
            mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
            if verbose:
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

    if verbose:
        print("\nBest classifier is " + get_classifier_name(best_clf) + "/" + str(best_ratio) +
              " with MCC = " + str(best_metrics["MCC"]))

    # Scores of the SPROUT wrapper
    det_dict = {"binary classifier": get_classifier_name(best_clf),
                "train data size": len(y_train),
                "train data features": numpy.asarray(features),
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

    if hasattr(best_clf, "feature_importances_"):
        f_imp = dict(zip(numpy.asarray(features), best_clf.feature_importances_))
    elif hasattr(best_clf, "coef_"):
        fi_scores = abs(best_clf.coef_[0])
        f_imp = dict(zip(numpy.asarray(features), fi_scores / sum(fi_scores)))
    else:
        print("No feature importance can be computed for " + get_classifier_name(best_clf))
        f_imp = {}

    return best_clf, f_imp, det_dict


def sample_adj_data(x, y, ratio: float):
    """
    Samples data to support training the binary adjudicator. Sets a given ration between 0 and 1 instances
    :param x: data
    :param y: labels
    :param ratio: ratio inbetween (0, 1)
    :return: a couple data-labels
    """
    # Creating DataFrame
    df = pandas.DataFrame(x.copy())
    df["label"] = y
    normal_frame = df.loc[df["label"] == 0]
    misc_frame = df.loc[df["label"] == 1]

    # Scaling to 'ratio'
    df_ratio = len(misc_frame.index) / len(normal_frame.index)
    if df_ratio < ratio:
        normal_frame = normal_frame.sample(frac=(df_ratio / (2 * ratio)))
    df = pandas.concat([normal_frame, misc_frame])
    df = df.sample(frac=1.0)

    return df.drop(["label"], axis=1).to_numpy(), df["label"].to_numpy()
