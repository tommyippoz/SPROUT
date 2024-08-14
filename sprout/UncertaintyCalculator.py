import copy
import random
import warnings
from collections import Counter

import joblib
import numpy
import numpy as np
import pandas
import pandas as pd
import pyod.models.base
import scipy.stats
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from pyod.models.copod import COPOD
from scipy.stats import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import *
from sprout.classifiers.AutoEncoder import DeepAutoEncoder, SingleAutoEncoder, SingleSparseAutoEncoder
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.general_utils import current_ms, get_full_class_name
from sprout.utils.sprout_utils import predictions_variability


class UncertaintyCalculator:
    """
    Abstract Class for uncertainty calculators. Methods to be overridden are uncertainty_strategy_name and uncertainty_scores
    """

    def uncertainty_calculator_name(self):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        """
        pass

    def full_uncertainty_calculator_name(self):
        """
        Returns the extended name of the strategy to calculate uncertainty score (as string)
        """
        return self.uncertainty_calculator_name()

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return None

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        pass


class MaxProbUncertainty(UncertaintyCalculator):
    """
    Computes uncertainty via Maximum probability assigned to a class for a given data point.
    Higher probability means high uncertainty / confidence
    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        if proba_array is None:
            proba_array = classifier.predict_proba(feature_values_array)
        max_p = numpy.max(proba_array, axis=1)
        return np.asarray(max_p)

    def uncertainty_calculator_name(self):
        return 'MaxProb Calculator'


class EntropyUncertainty(UncertaintyCalculator):
    """
    Computes uncertainty via Entropy of the probability array for a given data point.
    Higher entropy means low uncertainty / confidence
    """

    def __init__(self, norm=2):
        """
        Constructor Method
        :param norm: number of classes for normalization process
        """
        norm_array = np.full(norm, 1 / norm)
        self.normalization = (-norm_array * np.log2(norm_array)).sum()

    def uncertainty_score(self, proba):
        """
        Returns the entropy for a given prediction array
        :param proba: the probability array assigned by the algorithm to the data point
        :return: entropy score in the range [0, 1]
        """
        val = np.delete(proba, np.where(proba == 0))
        p = val / val.sum()
        entropy = (-p * np.log2(p)).sum()
        return (self.normalization - entropy) / self.normalization

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        uncertainty = []
        if not isinstance(feature_values_array, DataLoader):
            if not isinstance(feature_values_array, np.ndarray):
                feature_values_array = feature_values_array.to_numpy()
        if len(feature_values_array) == len(proba_array):
            uncertainty = [self.uncertainty_score(proba_array[i]) for i in range(0, len(proba_array))]
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(uncertainty)

    def uncertainty_calculator_name(self):
        return 'Entropy Calculator'


class NeighborsUncertainty(UncertaintyCalculator):
    """
    Computes uncertainty via Agreement with label predictions of neighbours.
    Reports both on the uncertainty and on the details for the neighbours.
    """

    def __init__(self, x_train, y_train, k, labels):
        self.x_train = x_train
        self.y_train = y_train
        try:
            self.n_neighbors = int(k)
        except:
            self.n_neighbors = 19
        self.labels = labels

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"n_neighbors": self.n_neighbors}

    def uncertainty_calculator_name(self):
        return 'uncertainty calculator on ' + str(self.n_neighbors) + ' Neighbors'

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Computes uncertainty by predicting the labels for the k-NN of each data point.
        uncertainty score ranges from 0 (complete disagreement) to 1 (complete agreement)
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :param classifier: the classifier used for classification
        :return: dictionary of two arrays: uncertainty and Detail
        """
        start_time = current_ms()
        print("Starting kNN search ...")
        near_neighbors = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          algorithm='kd_tree',
                                          n_jobs=-1).fit(self.x_train)
        distances, indices = near_neighbors.kneighbors(feature_values_array)
        print("kNN Search completed in " + str(current_ms() - start_time) + " ms")
        train_proba = np.asarray(classifier.predict_proba(self.x_train))
        train_classes = numpy.argmax(train_proba, axis=1)
        if proba_array is None or proba_array.shape[0] != feature_values_array.shape[0]:
            proba_array = np.asarray(classifier.predict_proba(feature_values_array))
        predict_classes = numpy.argmax(proba_array, axis=1)
        neighbour_agreement = [0.0 for i in range(len(feature_values_array))]
        neighbour_uncertainty = [0.0 for i in range(len(feature_values_array))]
        neighbour_c = [0 for i in range(len(feature_values_array))]
        for i in tqdm(range(len(feature_values_array))):
            predict_neighbours = train_classes[indices[i]]
            agreements = (predict_neighbours == predict_classes[i]).sum()
            neighbour_agreement[i] = agreements / len(predict_neighbours)
            neighbour_uncertainty[i] = predictions_variability(train_proba[indices[i]])
            neighbour_c[i] = Counter(list(map(lambda x: self.labels[x], predict_neighbours))).most_common()
        return {"agreement": neighbour_agreement, "uncertainty": neighbour_uncertainty, "Detail": neighbour_c}


class ExternalSupervisedUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that runs an external classifer and calculates its confidence in the result
    """

    def __init__(self, del_clf, x_train, y_train, norm=2, unc_measure='entropy'):
        self.del_clf = del_clf
        if unc_measure == 'entropy':
            self.u_measure = EntropyUncertainty(norm)
        else:
            self.u_measure = MaxProbUncertainty()
            unc_measure = 'max_prob'
        self.unc_measure = unc_measure
        if x_train is not None and y_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            self.del_clf.fit(x_train, y_train)
            print("[ExternalSupuncertainty] Fitting of '" + get_classifier_name(del_clf) + "' Completed")
        else:
            print("[ExternalSupuncertainty] Unable to train the supervised classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        joblib.dump(self.del_clf, main_folder + tag + "_del_clf.joblib", compress=9)
        return {"del_clf": get_full_class_name(self.del_clf.__class__), "unc_measure": self.unc_measure}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        return self.u_measure.uncertainty_scores(feature_values_array,
                                                 self.del_clf.predict_proba(feature_values_array),
                                                 self.del_clf)

    def uncertainty_calculator_name(self):
        return 'External Supervised Calculator (' + get_classifier_name(self.del_clf) + '/' \
               + str(self.unc_measure) + ')'


class ExternalUnsupervisedUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that runs an external classifer and calculates its confidence in the result
    """

    def __init__(self, del_clf, x_train, norm=2, unc_measure='entropy'):
        self.del_clf = del_clf
        if unc_measure == 'entropy':
            self.u_measure = EntropyUncertainty(norm)
        else:
            self.u_measure = MaxProbUncertainty()
            unc_measure = 'max_prob'
        self.unc_measure = unc_measure
        if x_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            self.del_clf.fit(x_train)
            print("[ExternalUnsuncertainty] Fitting of '" + get_classifier_name(del_clf) + "' Completed")
        else:
            print("[ExternalUnsuncertainty] Unable to train the supervised classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        joblib.dump(self.del_clf, main_folder + tag + "_del_clf.joblib", compress=9)
        return {"del_clf": get_full_class_name(self.del_clf.__class__), "unc_measure": self.unc_measure}

    def unsupervised_predict_proba(self, test_features):
        proba = self.del_clf.predict_proba(test_features)
        pred = self.del_clf.predict(test_features)
        for i in range(len(pred)):
            min_p = min(proba[i])
            max_p = max(proba[i])
            proba[i][pred[i]] = max_p
            proba[i][1 - pred[i]] = min_p
        return proba

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        if isinstance(classifier, pyod.models.base.BaseDetector):
            return self.u_measure.uncertainty_scores(feature_values_array,
                                                     self.unsupervised_predict_proba(feature_values_array),
                                                     self.del_clf)
        else:
            return self.u_measure.uncertainty_scores(feature_values_array,
                                                     self.del_clf.predict_proba(feature_values_array),
                                                     self.del_clf)

    def uncertainty_calculator_name(self):
        return 'External Unsupervised Calculator (' + get_classifier_name(self.del_clf) + '/' \
               + str(self.unc_measure) + ')'


class CombinedUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, del_clf, x_train, y_train=None, norm=2):
        self.del_clf = del_clf
        self.u_measure = EntropyUncertainty(norm)
        if x_train is not None:
            if not isinstance(x_train, DataLoader):
                if isinstance(x_train, pandas.DataFrame):
                    x_train = x_train.to_numpy()
            start_time = current_ms()

            if isinstance(self.del_clf, pyod.models.base.BaseDetector):
                self.del_clf.fit(x_train)
            else:
                if not isinstance(x_train, DataLoader):
                    self.del_clf.fit(x_train, y_train)
                else:
                    self.del_clf.fit(x_train)
            print("[Combineduncertainty] Fitting of '" + get_classifier_name(del_clf) + "' Completed in " +
                  str(current_ms() - start_time) + " ms")
        else:
            print("[CombinedUncertainty] Unable to train combined classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        joblib.dump(self.del_clf, main_folder + tag + "_del_clf.joblib", compress=9)
        return {"del_clf": get_full_class_name(self.del_clf.__class__)}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the combined uncertainty calculated using the main classifier plus the additional classifier
        Score ranges from
            -1 (complete and strong disagreement between the two classifiers) -> low confidence
        to
            1, which represents the complete agreement between the two classifiers and thus high confidence
        a score of 0 represents a very uncertain prediction
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        if isinstance(feature_values_array, np.ndarray):
            if proba_array is None or proba_array.shape[0] != feature_values_array.shape[0]:
                pred = classifier.predict(feature_values_array)
            else:
                pred = numpy.argmax(proba_array, axis=1)
        else:
            if proba_array is None:
                pred = classifier.predict(feature_values_array)
            else:
                pred = numpy.argmax(proba_array, axis=1)
        custom_pred = self.del_clf.predict_proba(feature_values_array)
        other_pred = self.del_clf.predict(feature_values_array)
        entropy = self.u_measure.uncertainty_scores(feature_values_array, proba_array, classifier)
        other_entropy = self.u_measure.uncertainty_scores(feature_values_array,
                                                          self.del_clf.predict_proba(feature_values_array),
                                                          self.del_clf)
        return np.where(pred == other_pred, (entropy + other_entropy) / 2, -(entropy + other_entropy) / 2)

    def uncertainty_calculator_name(self):
        return 'Combined Calculator (' + get_classifier_name(self.del_clf) + ')'


class MultiCombinedUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, clf_set, x_train, y_train=None, norm=2):
        self.uncertainty_set = []
        self.tag = ""
        start_time = current_ms()
        for clf in clf_set:
            self.uncertainty_set.append(CombinedUncertainty(clf, x_train, y_train, norm))
            self.tag = self.tag + get_classifier_name(clf)[0] + get_classifier_name(clf)[-1]
        self.tag = str(len(self.uncertainty_set)) + " - " + self.tag
        print("[MultiCombineduncertainty] Fitting of " + str(len(clf_set)) + " classifiers completed in "
              + str(current_ms() - start_time) + " ms")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        i = 1
        clf_names = []
        for uc in self.uncertainty_set:
            clf = uc.del_clf
            joblib.dump(clf, main_folder + tag + "_del_clf_" + str(i) + ".joblib", compress=9)
            clf_names.append(get_full_class_name(clf.__class__))
            i = i + 1
        return {"del_clfs": clf_names}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the combined uncertainty averaged over many combined classifiers
        Score ranges from
            -1 (complete and strong disagreement between the two classifiers) -> low confidence
        to
            1, which represents the complete agreement between the two classifiers and thus high confidence
        a score of 0 represents a very uncertain prediction
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        multi_uncertainty = []
        for combined_uncertainty in self.uncertainty_set:
            multi_uncertainty.append(
                combined_uncertainty.uncertainty_scores(feature_values_array, proba_array, classifier))
        return numpy.average(numpy.asarray(multi_uncertainty), axis=0)

    def uncertainty_calculator_name(self):
        return 'Multiple Combined Calculator (' + str(self.tag) + ' classifiers)'


class AgreementUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that measures agreement between a set of classifiers
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, clf_set, x_train, y_train=None):
        self.clfs = []
        self.tag = ""
        start_time = current_ms()
        for clf in clf_set:
            if x_train is not None:
                if isinstance(x_train, pandas.DataFrame):
                    x_train = x_train.to_numpy()
                try:
                    if isinstance(clf, pyod.models.base.BaseDetector):
                        clf.fit(x_train)
                    else:
                        clf.fit(x_train, y_train)
                except:
                    print("Classifier '" + get_classifier_name(clf) + "' did not train correctly")
            else:
                print("Classifier '" + get_classifier_name(clf) + "' did not train correctly - no data available")
            self.clfs.append(clf)
            self.tag = self.tag + get_classifier_name(clf)[0] + get_classifier_name(clf)[-1]
        self.tag = str(len(self.clfs)) + " - " + self.tag
        print("[Agreementuncertainty] Fitting of " + str(len(clf_set)) + " classifiers completed in "
              + str(current_ms() - start_time) + " ms")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        i = 1
        clf_names = []
        for clf in self.clfs:
            joblib.dump(clf, main_folder + tag + "_clf_" + str(i) + ".joblib", compress=9)
            clf_names.append(get_full_class_name(clf.__class__))
            i = i + 1
        return {"clfs": clf_names}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the combined uncertainty averaged over many combined classifiers
        Score ranges from
            0 (complete and strong disagreement between the classifiers) -> low confidence
        to
            1, which represents the complete agreement between the classifiers and thus high confidence
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        multi_uncertainty = []
        for clf_model in self.clfs:
            try:
                predictions = numpy.asarray(clf_model.predict(feature_values_array))
                multi_uncertainty.append(predictions)
            except:
                print("Classifier '" + get_classifier_name(clf_model) + "' cannot be used for prediction")
        multi_uncertainty = numpy.asarray(multi_uncertainty)
        mode_value = stats.mode(multi_uncertainty)
        scores = numpy.where(multi_uncertainty == mode_value, 1, 0)
        return numpy.average(scores, axis=1)[0]

    def uncertainty_calculator_name(self):
        return 'Agreement Calculator (' + str(self.tag) + ' classifiers)'


class ConfidenceInterval(UncertaintyCalculator):
    """
    Defines a uncertainty measure that calculates confidence intervals to derive uncertainty
    """

    def __init__(self, x_train, y_train=None, conf_level=0.9999):
        try:
            self.confidence_level = float(conf_level)
        except:
            self.confidence_level = 0.9999
        self.intervals_min = {}
        self.intervals_max = {}
        self.labels = numpy.unique(y_train)

        if x_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_data = x_train.to_numpy()
            else:
                x_data = x_train
            if y_train is None:
                self.interval_type = 'uns'
                intervals = []
                for i in range(0, len(x_data[0])):
                    feature = x_data[:, i]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        intervals.append(scipy.stats.t.interval(self.confidence_level,
                                                                len(feature) - 1,
                                                                loc=np.median(feature),
                                                                scale=scipy.stats.sem(feature)))
                intervals = numpy.asarray(intervals)
                self.intervals_min = numpy.asarray(intervals[:, 0])
                self.intervals_max = numpy.asarray(intervals[:, 1])
            else:
                self.interval_type = 'sup'
                for label in self.labels:
                    intervals = []
                    data = x_data[y_train == label, :]
                    for i in range(0, len(x_data[0])):
                        feature = data[:, i]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            intervals.append(scipy.stats.t.interval(self.confidence_level,
                                                                    len(feature) - 1,
                                                                    loc=np.median(feature),
                                                                    scale=scipy.stats.sem(feature)))
                    intervals = numpy.asarray(intervals)
                    self.intervals_min[label] = numpy.asarray(intervals[:, 0])
                    self.intervals_max[label] = numpy.asarray(intervals[:, 1])
        else:
            self.interval_type = 'none'
            print("Unable to derive confidence intervals - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"confidence_level": self.confidence_level}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute uncertainty score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        uncertainty = []
        # predicted_labels = numpy.argmax(proba_array, axis=1)
        predicted_labels = proba_array
        if isinstance(feature_values_array, pandas.DataFrame):
            feature_values_array = feature_values_array.to_numpy()
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                if self.interval_type == 'sup':
                    in_left = (self.intervals_min[predicted_labels[i]] <= feature_values_array[i])
                    in_right = (feature_values_array[i] <= self.intervals_max[predicted_labels[i]])
                else:
                    in_left = (self.intervals_min <= feature_values_array[i])
                    in_right = (feature_values_array[i] <= self.intervals_max)
                uncertainty.append(numpy.average(in_left * in_right))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(uncertainty)

    def uncertainty_calculator_name(self):
        return 'Confidence Interval (' + str(self.confidence_level) + '/' + str(self.interval_type) + ')'


class ProximityUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that creates artificial neighbours of a data point
    and checks if the classifier has a unified answer to all of those data points
    """

    def __init__(self, x_train, artificial_points: int = 10, range_wideness: float = 0.1, weighted: bool = False):
        self.n_artificial = int(artificial_points)
        self.range = float(range_wideness)
        self.weighted = weighted

        if x_train is not None:
            start_time = current_ms()
            if isinstance(x_train, pd.DataFrame):
                x_train = x_train.to_numpy()
            self.stds = numpy.std(x_train, axis=0)
            print("Proximity Uncertainty initialized in " + str(current_ms() - start_time) + " ms")
        else:
            print("Proximity Uncertainty failed to initialize - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"artificial_points": self.n_artificial, "weighted": self.weighted, "range": self.range}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the uncertainty after executing a given amount of simulations around the feature values
        Score ranges from -1 (likely to be misclassification) to 1 (likely to be correct classification)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        predicted_classes = numpy.argmax(proba_array, axis=1)
        if isinstance(feature_values_array, pd.DataFrame):
            feature_values_array = feature_values_array.to_numpy()

        # Generating MC Artificial inputs
        mc_x = []
        rng = np.random.default_rng()
        for i in range(len(feature_values_array)):
            features = feature_values_array[i]
            mc_x.extend(features + self.range * self.stds *
                        rng.uniform(low=-1, high=1, size=(self.n_artificial, len(features))))
        mc_x = np.array(mc_x)

        # Calculating predictions
        mc_predict = classifier.predict(mc_x)

        # Calculating Uncertainty
        uncertainty = []
        for i in range(len(feature_values_array)):
            mc_preds = mc_predict[i * self.n_artificial:(i + 1) * self.n_artificial]
            if self.weighted:
                artificial_features = mc_x[i * self.n_artificial:(i + 1) * self.n_artificial, :]
                distances = [numpy.linalg.norm(af - feature_values_array[i]) for af in artificial_features]
                relevance = (max(distances) - distances) + 0.1 * max(distances)
                relevance = relevance / sum(relevance)
                score = sum((mc_preds == predicted_classes[i]) * relevance)
            else:
                score = sum(mc_preds == predicted_classes[i]) / self.n_artificial
            uncertainty.append(score)

        return np.asarray(uncertainty)

    def uncertainty_calculator_name(self):
        return 'Proximity Uncertainty (' + str(self.n_artificial) + '/' + str(self.range) \
               + ('/W' if self.weighted else '') + ')'


class FeatureBaggingUncertainty(UncertaintyCalculator):
    """
    Defines a uncertainty measure that uses a Monte Carlo simulation for each class
    """

    def __init__(self, x_train, y_train, n_baggers=10, bag_type='sup'):
        self.feature_sets = []
        self.classifiers = []
        try:
            self.n_baggers = int(n_baggers)
        except:
            self.n_baggers = 10

        if x_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            n_features = x_train.shape[1]

            if n_features < 20:
                bag_rate = 0.8
            elif n_features < 50:
                bag_rate = 0.7
            elif n_features < 100:
                bag_rate = 0.6
            else:
                bag_rate = 0.5
            bag_features = int(n_features * bag_rate)

            for i in tqdm(range(self.n_baggers), "Building Feature Baggers"):
                fs = random.sample(range(n_features), bag_features)
                fs.sort()
                self.feature_sets.append(fs)
                if bag_type == 'uns':
                    classifier = COPOD()
                    classifier.fit(x_train[:, fs])
                else:
                    classifier = DecisionTreeClassifier()
                    classifier.fit(x_train[:, fs], y_train)
                    bag_type = 'sup'
                self.classifiers.append(classifier)

        else:
            print("Unable to build feature baggers - no data available")

        self.bag_type = bag_type

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"n_baggers": self.n_baggers, "bag_type": self.bag_type}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the uncertainty after executing a given amount of simulations around the feature values
        Score ranges from 0 (no agreement) to 1 (full agreement)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        predicted_classes = numpy.argmax(proba_array, axis=1)
        if isinstance(feature_values_array, pd.DataFrame):
            feature_values_array = feature_values_array.to_numpy()

        # Testing with all classifiers
        fs_pred = [self.classifiers[i].predict(feature_values_array[:, self.feature_sets[i]])
                   for i in range(len(self.feature_sets))]
        fs_pred = numpy.array(fs_pred).transpose()

        # Calculating Uncertainty
        uncertainty = [sum(fs_pred[i] == predicted_classes[i]) / len(fs_pred[i])
                       for i in range(len(feature_values_array))]

        return np.asarray(uncertainty)

    def uncertainty_calculator_name(self):
        return 'FeatureBagging Uncertainty (' + str(self.n_baggers) + '/' + str(self.bag_type) + ')'


class ConfidenceBaggingUncertainty(CombinedUncertainty):
    """
    Defines a uncertainty measure that creates a bagging/boosting meta learner using a generic clf as a base estimator
    """

    def __init__(self, clf, x_train, y_train=None, n_base: int = 10, max_features: float = 0.7, sampling_ratio: float = 0.7,
                 perc_decisors: float = None, n_decisors: int = None, n_classes: int = 2):
        super().__init__(ConfidenceBagging(clf, n_base, max_features, sampling_ratio, perc_decisors, n_decisors),
                         x_train, y_train, n_classes)
        self.n_classes = n_classes

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"clf": self.del_clf.clf.__class__.__name__,
                "n_base": self.del_clf.n_base,
                "max_features": self.del_clf.max_features,
                "sampling_ratio": self.del_clf.sampling_ratio,
                "n_decisors": self.del_clf.n_decisors,
                "n_classes": self.n_classes}

    def uncertainty_calculator_name(self):
        return "ConfidenceBagger(" + str(self.del_clf.n_base) + "-" + str(self.del_clf.n_decisors) + "-" + \
               str(self.del_clf.max_features) + "-" + str(self.del_clf.sampling_ratio) + ")"


class ConfidenceBoostingUncertainty(CombinedUncertainty):
    """
    Defines a uncertainty measure that creates a bagging/boosting meta learner using a generic clf as a base estimator
    """

    def __init__(self, clf, x_train, y_train=None, n_base: int = 10, learning_rate: float = None, sampling_ratio: float = 0.5,
                 contamination: float = None, conf_thr: float = 0.8, n_classes: int = 2):
        super().__init__(ConfidenceBoosting(clf, n_base, learning_rate, sampling_ratio, contamination, conf_thr),
                         x_train, y_train, n_classes)
        self.n_classes = n_classes

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"clf": self.del_clf.clf.__class__.__name__,
                "n_base": self.del_clf.n_base,
                "learning_rate": self.del_clf.learning_rate,
                "sampling_ratio": self.del_clf.sampling_ratio,
                "contamination": self.del_clf.contamination,
                "conf_thr": self.del_clf.conf_thr,
                "n_classes": self.n_classes}

    def uncertainty_calculator_name(self):
        return "ConfidenceBooster(" + str(self.del_clf.n_base) + "-" + str(self.del_clf.conf_thr) + "-" + \
               str(self.del_clf.learning_rate) + "-" + str(self.del_clf.sampling_ratio) + ")"


class ReconstructionLoss(UncertaintyCalculator):
    """
    Defines a uncertainty measure that uses the reconstruction error of an autoencoder as uncertainty measure
    """

    def __init__(self, x_train, enc_tag:str = 'simple'):
        self.ae = None
        self.enc_tag = enc_tag

        if x_train is not None:
            bottleneck = int(x_train.shape[1] / 4) if x_train.shape[1] > 8 else 2
            if enc_tag == 'deep':
                self.ae = DeepAutoEncoder(x_train.shape[1], bottleneck)
            elif enc_tag == 'sparse':
                self.ae = SingleSparseAutoEncoder(x_train.shape[1], bottleneck)
            else:
                self.enc_tag = 'simple'
                self.ae = SingleAutoEncoder(x_train.shape[1], bottleneck)
            norm_tr = copy.deepcopy(x_train)
            if isinstance(norm_tr, pandas.DataFrame):
                norm_tr = norm_tr.to_numpy()
            self.maxmin = []
            for i in range(0, norm_tr.shape[1]):
                self.maxmin.append({'min': min(norm_tr[:, i]), 'max': max(norm_tr[:, i])})
                if self.maxmin[i]['max'] - self.maxmin[i]['min'] != 0:
                    norm_tr[:, i] = (norm_tr[:, i] - self.maxmin[i]['min']) / \
                                    (self.maxmin[i]['max'] - self.maxmin[i]['min'])
            self.ae.fit(norm_tr, 50, 256, verbose=0)
        else:
            print("Unable to build autoencoder to compute loss - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"enc_tag": self.enc_tag}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the uncertainty after executing a given amount of simulations around the feature values
        Score ranges from 0 (no agreement) to 1 (full agreement)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        norm_te = copy.deepcopy(feature_values_array)
        if isinstance(norm_te, pandas.DataFrame):
            norm_te = norm_te.to_numpy()
        for i in range(0, len(self.maxmin)):
            if self.maxmin[i]['max'] - self.maxmin[i]['min'] != 0:
                norm_te[:, i] = (norm_te[:, i] - self.maxmin[i]['min']) / \
                                (self.maxmin[i]['max'] - self.maxmin[i]['min'])
        try:
            decoded_tr, loss_tr = self.ae.predict(norm_te)
        except:
            loss_tr = 0
        return np.asarray(loss_tr)

    def uncertainty_calculator_name(self):
        return 'AutoEncoder Loss (' + str(self.enc_tag) + ')'

class ReconstructionLoss(UncertaintyCalculator):
    """
    Defines an uncertainty measure that uses the reconstruction error of an autoencoder as uncertainty measure.
    """

    def __init__(self, dataloader, enc_tag: str = 'conv'):
        self.ae = None
        self.enc_tag = enc_tag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Assuming DataLoader provides batches of data
        x_train, _ = next(iter(dataloader))  # Get one batch to initialize autoencoder
        if x_train is not None:
            _, channels, _, _ = x_train.shape
            # print("Channels is the Function ",channels)
            if enc_tag == 'conv':
                self.ae = ConvAutoEncoder(channels).to(self.device)
            else:
                raise ValueError(f"Unsupported encoder type: {enc_tag}")

            # Normalize data
            self.dataloader = dataloader
            self.ae.train()
            optimizer = optim.Adam(self.ae.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            epochs = 1

            for epoch in range(epochs):  # 50 epochs
                epoch_loss = 0.0  # Initialize epoch_loss
                for batch, _ in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.ae(batch)
                    loss = criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate uncertainty score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"enc_tag": self.enc_tag}

    def uncertainty_scores(self, dataloader, proba_array, classifier):
        """
        Returns the uncertainty after executing a given amount of simulations around the feature values.
        Score ranges from 0 (no agreement) to 1 (full agreement).

        :param classifier: the classifier used for classification
        :param dataloader: DataLoader for the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of uncertainty scores
        """
        all_losses = []
        self.ae.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch,_ in tqdm.tqdm(dataloader, desc="Evaluating"):
                # if isinstance(batch, tuple) and len(batch) == 2:
                #     batch, _ = batch  # Unpack if dataloader returns (data, label)
                batch = batch.to(self.device)
                outputs = self.ae(batch)
                loss = criterion(outputs, batch)
                all_losses.append(loss.cpu().numpy())
        return np.asarray(all_losses)

    def uncertainty_calculator_name(self):
        return 'AutoEncoder Loss (' + str(self.enc_tag) + ')'