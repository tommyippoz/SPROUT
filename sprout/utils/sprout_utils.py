import csv
import os.path

import numpy
import numpy as np
import scipy
import sklearn


def build_SPROUT_dataset(x_test, y_proba, y_pred, y_test, label_tags):
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


def correlations(trust_df, corr_tag="INFO", print_output=True):
    """
    Returns Correlations of Trust Measures w.r.t. misclassifications of the classifier
    :param print_output: True if results have to be printed on screen
    :param corr_tag: tag that specified the type of correlation analysis
    :param trust_df: dataframe containing trust measures and label
    """
    corr_dict = {}
    label = trust_df["is_misclassification"].to_numpy()
    if print_output:
        print("Importance of each uncertainty measure with SPROUT behavior")
    for feature_name in trust_df.columns:
        if feature_name not in ["true_label", "predicted_label", "is_misclassification", "probabilities"]:
            feature_values = trust_df[feature_name]
            if len(feature_values) > 0 and not isinstance(feature_values[0], list):
                corr_dict[feature_name] = compute_correlation(feature_values.to_numpy(), label, corr_tag)
                if print_output:
                    print("'" + corr_tag + "' Correlation of '" + feature_name + "' with label: " +
                          str(corr_dict[feature_name]))
    return corr_dict


def compute_correlation(feature, label, corr_tag="INFO"):
    """
    Computes correlation (double value) between two arrays
    :param feature: first array
    :param label:  second array (reference)
    :param corr_tag: type of correlation analysis
    :return: double value
    """
    if (feature is not None) & (label is not None) & (len(feature) == len(label)):
        if corr_tag is not None:
            if corr_tag.upper() in ["R2", "R-SQUARED", "R-SQ"]:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(feature, label)
                return r_value ** 2
            elif corr_tag.upper() in ["P", "PEARSON", "PEARSON CORRELATION"]:
                return abs(scipy.stats.pearsonr(feature, label)[0])
            elif corr_tag.upper() in ["SP", "SPEARMAN"]:
                return abs(scipy.stats.spearmanr(feature, label)[0])
            elif corr_tag.upper() in ["COS", "COSINE"]:
                return 1 - scipy.spatial.distance.cosine(feature, label)
            elif corr_tag.upper() in ["CHI", "CHI2", "CHI-2", "CHI-SQUARED", "CHI-SQ"]:
                scaled_feat_values = sklearn.preprocessing.MinMaxScaler().fit_transform(feature.reshape(-1, 1))
                return sklearn.feature_selection.chi2(scaled_feat_values, label)[0][0]
            elif corr_tag.upper() in ["INFO", "MUTUAL INFORMATION", "MI", "MUTUALINFO"]:
                return sklearn.feature_selection.mutual_info_classif(feature.reshape(-1, 1), label)[0]
        print("Unable to recognize correlation tag, default R-Squared will be used")
        return compute_correlation("INFO", feature, label)
    else:
        print("Features and Label are not arrays of the same size")
        return 0.0


def read_calculators(model_folder):
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


def predictions_variability(predictions):
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


def compute_omission_metrics(y_true, y_wrapper, y_clf, reject_tag=None):
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
    met_dict['phi_c'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf == y_true), 1, 0))/len(y_true)
    met_dict['phi_m'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf != y_true), 1, 0)) / len(y_true)
    met_dict['eps_gain'] = 0 if met_dict['eps'] == 0 else (met_dict['eps'] - met_dict['eps_w']) / met_dict['eps']
    met_dict['phi_m_ratio'] = 0 if met_dict['phi'] == 0 else met_dict['phi_m'] / met_dict['phi']
    met_dict['overall'] = 2*met_dict['eps_gain']*met_dict['phi_m_ratio']/(met_dict['eps_gain'] + met_dict['phi_m_ratio'])
    return met_dict
