import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn as sk

from Classifier import Classifier


def build_QUAIL_dataset(y_proba, y_pred, y_test, label_tags):
    """
    Prepares DataFrame to output SPROUT results
    :param y_proba: probabilities assigned by the classifier
    :param y_pred: predictions (classes) of the classifier
    :param y_test: labels of the test set (ground truth)
    :param label_tags: Names of the classes
    :return: a DataFrame with 4 columns
    """
    out_df = pd.DataFrame()
    out_df['true_label'] = list(map(lambda x: label_tags[x], y_test))
    out_df['predicted_label'] = list(map(lambda x: label_tags[x], y_pred))
    out_df['is_misclassification'] = np.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
    out_df['probabilities'] = [np.array2string(y_proba[i], separator=";") for i in range(len(y_proba))]
    a = y_proba[0]
    return out_df


def build_classifier(classifier, x_train, y_train, x_test, y_test, verbose=True):
    """
    Builds and Exercises a Classifier
    :param classifier: classifier object
    :param x_train: train features
    :param y_train: train label
    :param x_test: test features
    :param y_test: test label
    :param verbose: True if there should be console output
    :return: probabilities and predictions of the classifier
    """
    if verbose:
        print("\nBuilding classifier: " + classifier.classifier_name() + "\n")

    # Fitting classifier
    start_ms = utils.current_ms()
    classifier.fit(x_train, y_train)
    train_ms = utils.current_ms()

    # Test features have to be a numpy array
    if not isinstance(x_test, np.ndarray):
        x_test = x_test.to_numpy()

    # Predicting labels
    y_pred = classifier.predict(x_test)
    test_time = utils.current_ms() - train_ms

    # Predicting probabilities
    y_proba = classifier.predict_proba(x_test)
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.to_numpy()

    if verbose:
        print(classifier.classifier_name() + " train/test in " + str(train_ms - start_ms) + "/" +
              str(test_time) + " ms with Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))

    return y_proba, y_pred


def get_classifier_name(clf):
    if isinstance(clf, Classifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def get_feature_importance(clf):
    if isinstance(clf, Classifier):
        return clf.feature_importances()
    else:
        return clf.feature_importances_


def correlations(trust_df, corr_tag="R2", print_output=True):
    """
    Returns Correlations of Trust Measures w.r.t. misclassifications of the classifier
    :param print_output: True if results have to be printed on screen
    :param corr_tag: tag that specified the type of correlation analysis
    :param trust_df: dataframe containing trust measures and label
    """
    corr_dict = {}
    label = trust_df["is_misclassification"]
    for feature_name in trust_df.columns:
        if feature_name not in ["true_label", "predicted_label", "is_misclassification", "probabilities"]:
            corr_dict[feature_name] = compute_correlation(trust_df[feature_name], label, corr_tag)
            if print_output:
                print("'" + corr_tag + "' Correlation of '" + feature_name + "' with label: " +
                      str(corr_dict[feature_name]))
    return corr_dict


def compute_correlation(feature, label, corr_tag="R2"):
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
        return compute_correlation("R2", feature, label)
    else:
        print("Features and Label are not arrays of the same size")
        return 0.0
