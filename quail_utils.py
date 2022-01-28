import numpy as np
import pandas as pd
import sklearn as sk

import Classifier
import utils


def build_QUAIL_dataset(y_proba, y_pred, y_test, label_tags):
    """
    Prepares DataFrame to output QUAIL results
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
    out_df['probabilities'] = [y_proba[i] for i in range(len(y_proba))]
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
    if isinstance(clf, Classifier.Classifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__

    pass
