import copy
import random
import warnings

import joblib
import numpy
import pandas
import pandas as pd
import pyod.models.base
import scipy.stats

import numpy as np
from pyod.models.copod import COPOD
from scipy.stats import stats

from sklearn.neighbors import NearestNeighbors
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from sprout.utils.AutoEncoder import DeepAutoEncoder, SingleAutoEncoder, SingleSparseAutoEncoder
from sprout.utils.sprout_utils import get_classifier_name
from sprout.utils.general_utils import current_ms, get_full_class_name


class UncertaintyCalculator:
    """
    Abstract Class for trust calculators. Methods to be overridden are trust_strategy_name and trust_scores
    """

    def uncertainty_calculator_name(self):
        """
        Returns the name of the strategy to calculate trust score (as string)
        """
        pass

    def full_uncertainty_calculator_name(self):
        """
        Returns the extended name of the strategy to calculate trust score (as string)
        """
        return self.uncertainty_calculator_name()

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return None

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        pass


class MaxProbUncertainty(UncertaintyCalculator):
    """
    Computes Trust via Maximum probability assigned to a class for a given data point.
    Higher probability means high trust / confidence
    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        if proba_array is None:
            proba_array = classifier.predict_proba(feature_values_array)
        max_p = numpy.max(proba_array, axis=1)
        return np.asarray(max_p)

    def uncertainty_calculator_name(self):
        return 'MaxProb Calculator'


class EntropyUncertainty(UncertaintyCalculator):
    """
    Computes Trust via Entropy of the probability array for a given data point.
    Higher entropy means low trust / confidence
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
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        trust = []
        if not isinstance(feature_values_array, np.ndarray):
            feature_values_array = feature_values_array.to_numpy()
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                trust.append(self.uncertainty_score(proba_array[i]))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(trust)

    def uncertainty_calculator_name(self):
        return 'Entropy Calculator'


class NeighborsUncertainty(UncertaintyCalculator):
    """
    Computes Trust via Agreement with label predictions of neighbours.
    Reports both on the trust and on the details for the neighbours.
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
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"n_neighbors": self.n_neighbors}

    def uncertainty_calculator_name(self):
        return 'Trust Calculator on ' + str(self.n_neighbors) + ' Neighbors'

    def uncertainty_scores(self, feature_values, proba, classifier):
        """
        Computes trust by predictng the labels for the k-NN of each data point.
        Trust score ranges from 0 (complete disagreement) to 1 (complete agreement)
        :param feature_values: the feature values of the data points in the test set
        :param proba: the probability arrays assigned by the algorithm to the data points
        :param classifier: the classifier used for classification
        :return: dictionary of two arrays: Trust and Detail
        """
        neighbour_trust = [0 for i in range(len(feature_values))]
        neighbour_c = [0 for i in range(len(feature_values))]
        start_time = current_ms()
        print("Starting kNN search ...")
        near_neighbors = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          algorithm='kd_tree',
                                          n_jobs=-1).fit(self.x_train)
        distances, indices = near_neighbors.kneighbors(feature_values)
        print("kNN Search completed in " + str(current_ms() - start_time) + " ms")
        train_classes = np.asarray(classifier.predict(self.x_train))
        predict_classes = np.asarray(classifier.predict(feature_values))
        for i in tqdm(range(len(feature_values))):
            predict_neighbours = train_classes[indices[i]]
            agreements = (predict_neighbours == predict_classes[i]).sum()
            neighbour_trust[i] = agreements / len(predict_neighbours)
            neighbour_c[i] = Counter(list(map(lambda x: self.labels[x], predict_neighbours))).most_common()
        return {"Trust": neighbour_trust, "Detail": neighbour_c}


class ExternalSupervisedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that runs an external classifer and calculates its confidence in the result
    """

    def __init__(self, del_clf, x_train, y_train, norm=2, unc_measure='entropy'):
        self.del_clf = del_clf
        if unc_measure == 'entropy':
            self.trust_measure = EntropyUncertainty(norm)
        else:
            self.trust_measure = MaxProbUncertainty()
            unc_measure = 'max_prob'
        self.unc_measure = unc_measure
        if x_train is not None and y_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            self.del_clf.fit(x_train, y_train)
            print("[ExternalSupTrust] Fitting of '" + get_classifier_name(del_clf) + "' Completed")
        else:
            print("[ExternalSupTrust] Unable to train the supervised classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        joblib.dump(self.del_clf, main_folder + tag + "_del_clf.joblib", compress=9)
        return {"del_clf": get_full_class_name(self.del_clf.__class__), "unc_measure": self.unc_measure}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        return self.trust_measure.uncertainty_scores(feature_values_array,
                                                     self.del_clf.predict_proba(feature_values_array),
                                                     self.del_clf)

    def uncertainty_calculator_name(self):
        return 'External Supervised Calculator (' + get_classifier_name(self.del_clf) + '/' \
               + str(self.unc_measure) + ')'


class ExternalUnsupervisedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that runs an external classifer and calculates its confidence in the result
    """

    def __init__(self, del_clf, x_train, norm=2, unc_measure='entropy'):
        self.del_clf = del_clf
        if unc_measure == 'entropy':
            self.trust_measure = EntropyUncertainty(norm)
        else:
            self.trust_measure = MaxProbUncertainty()
            unc_measure = 'max_prob'
        self.unc_measure = unc_measure
        if x_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            self.del_clf.fit(x_train)
            print("[ExternalUnsTrust] Fitting of '" + get_classifier_name(del_clf) + "' Completed")
        else:
            print("[ExternalUnsTrust] Unable to train the supervised classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
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
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        if isinstance(classifier, pyod.models.base.BaseDetector):
            return self.trust_measure.uncertainty_scores(feature_values_array,
                                                         self.unsupervised_predict_proba(feature_values_array),
                                                         self.del_clf)
        else:
            return self.trust_measure.uncertainty_scores(feature_values_array,
                                                         self.del_clf.predict_proba(feature_values_array),
                                                         self.del_clf)

    def uncertainty_calculator_name(self):
        return 'External Unsupervised Calculator (' + get_classifier_name(self.del_clf) + '/' \
               + str(self.unc_measure) + ')'


class CombinedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, del_clf, x_train, y_train=None, norm=2):
        self.del_clf = del_clf
        self.trust_measure = EntropyUncertainty(norm)
        if x_train is not None:
            if isinstance(x_train, pandas.DataFrame):
                x_train = x_train.to_numpy()
            start_time = current_ms()
            self.del_clf.fit(x_train, y_train)
            print("[CombinedTrust] Fitting of '" + get_classifier_name(del_clf) + "' Completed in " +
                  str(current_ms() - start_time) + " ms")
        else:
            print("[CombinedTrust] Unable to train combined classifier - no data available")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        joblib.dump(self.del_clf, main_folder + tag + "_del_clf.joblib", compress=9)
        return {"del_clf": get_full_class_name(self.del_clf.__class__)}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the combined trust calculated using the main classifier plus the additional classifier
        Score ranges from
            -1 (complete and strong disagreement between the two classifiers) -> low confidence
        to
            1, which represents the complete agreement between the two classifiers and thus high confidence
        a score of 0 represents a very uncertain prediction
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        pred = classifier.predict(feature_values_array)
        other_pred = self.del_clf.predict(feature_values_array)
        entropy = self.trust_measure.uncertainty_scores(feature_values_array, proba_array, classifier)
        other_entropy = self.trust_measure.uncertainty_scores(feature_values_array,
                                                              self.del_clf.predict_proba(feature_values_array),
                                                              self.del_clf)
        return np.where(pred == other_pred, (entropy + other_entropy) / 2, -(entropy + other_entropy) / 2)

    def uncertainty_calculator_name(self):
        return 'Combined Calculator (' + get_classifier_name(self.del_clf) + ')'


class MultiCombinedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, clf_set, x_train, y_train=None, norm=2):
        self.trust_set = []
        self.tag = ""
        start_time = current_ms()
        for clf in clf_set:
            self.trust_set.append(CombinedUncertainty(clf, x_train, y_train, norm))
            self.tag = self.tag + get_classifier_name(clf)[0] + get_classifier_name(clf)[-1]
        self.tag = str(len(self.trust_set)) + " - " + self.tag
        print("[MultiCombinedTrust] Fitting of " + str(len(clf_set)) + " classifiers completed in "
              + str(current_ms() - start_time) + " ms")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        i = 1
        clf_names = []
        for uc in self.trust_set:
            clf = uc.del_clf
            joblib.dump(clf, main_folder + tag + "_del_clf_" + str(i) + ".joblib", compress=9)
            clf_names.append(get_full_class_name(clf.__class__))
            i = i + 1
        return {"del_clfs": clf_names}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the combined trust averaged over many combined classifiers
        Score ranges from
            -1 (complete and strong disagreement between the two classifiers) -> low confidence
        to
            1, which represents the complete agreement between the two classifiers and thus high confidence
        a score of 0 represents a very uncertain prediction
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        multi_trust = np.zeros(len(feature_values_array))
        for combined_trust in self.trust_set:
            multi_trust = multi_trust + combined_trust.uncertainty_scores(feature_values_array, proba_array, classifier)
        return multi_trust / len(self.trust_set)

    def uncertainty_calculator_name(self):
        return 'Multiple Combined Calculator (' + str(self.tag) + ' classifiers)'


class AgreementUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that measures agreement between a set of classifiers
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
        print("[AgreementTrust] Fitting of " + str(len(clf_set)) + " classifiers completed in "
              + str(current_ms() - start_time) + " ms")

    def save_params(self, main_folder, tag):
        """
        Returns the name of the strategy to calculate trust score (as string)
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
        Returns the combined trust averaged over many combined classifiers
        Score ranges from
            0 (complete and strong disagreement between the classifiers) -> low confidence
        to
            1, which represents the complete agreement between the classifiers and thus high confidence
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        multi_trust = []
        for clf_model in self.clfs:
            try:
                predictions = numpy.asarray(clf_model.predict(feature_values_array))
                multi_trust.append(predictions)
            except:
                print("Classifier '" + get_classifier_name(clf_model) + "' cannot be used for prediction")
        multi_trust = numpy.asarray(multi_trust)
        mode_value = stats.mode(multi_trust)
        scores = numpy.where(multi_trust == mode_value, 1, 0)
        return numpy.average(scores, axis=1)[0]

    def uncertainty_calculator_name(self):
        return 'Agreement Calculator (' + str(self.tag) + ' classifiers)'


class ConfidenceInterval(UncertaintyCalculator):
    """
    Defines a trust strategy that calculates confidence intervals to derive trust
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
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"confidence_level": self.confidence_level}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        trust = []
        predicted_labels = numpy.argmax(proba_array, axis=1)
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
                trust.append(numpy.average(in_left * in_right))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(trust)

    def uncertainty_calculator_name(self):
        return 'Confidence Interval (' + str(self.confidence_level) + '/' + str(self.interval_type) + ')'


class ProximityUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that creates artificial neighbours of a data point
    and checks if the classifier has a unified answer to all of those data points
    """

    def __init__(self, x_train, artificial_points=10, range_wideness=0.1, weighted=False):
        try:
            self.n_artificial = int(artificial_points)
        except:
            self.n_artificial = 10
        try:
            self.range = float(range_wideness)
        except:
            self.range = 0.1
        try:
            self.weighted = weighted == "True"
        except:
            self.weighted = False

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
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"artificial_points": self.n_artificial, "weighted": self.weighted, "range": self.range}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the trust after executing a given amount of simulations around the feature values
        Score ranges from -1 (likely to be misclassification) to 1 (likely to be correct classification)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        predicted_classes = numpy.argmax(proba_array, axis=1)
        if isinstance(feature_values_array, pd.DataFrame):
            feature_values_array = feature_values_array.to_numpy()

        # Generating MC Artificial inputs
        mc_x = []
        for i in range(len(feature_values_array)):
            features = feature_values_array[i]
            mc_x.extend([[random.gauss(m, s) for m, s in zip(features, self.range * self.stds)]
                         for _ in range(self.n_artificial)])
        mc_x = np.array(mc_x)

        # Calculating predictions
        mc_predict = classifier.predict(mc_x)

        # Calculating Uncertainty
        trust = []
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
            trust.append(score)

        return np.asarray(trust)

    def uncertainty_calculator_name(self):
        return 'Proximity Uncertainty (' + str(self.n_artificial) + '/' + str(self.range) \
               + ('/W' if self.weighted else '') + ')'


class FeatureBagging(UncertaintyCalculator):
    """
    Defines a trust strategy that uses a Monte Carlo simulation for each class
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
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"n_baggers": self.n_baggers, "bag_type": self.bag_type}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the trust after executing a given amount of simulations around the feature values
        Score ranges from 0 (no agreement) to 1 (full agreement)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        predicted_classes = numpy.argmax(proba_array, axis=1)
        if isinstance(feature_values_array, pd.DataFrame):
            feature_values_array = feature_values_array.to_numpy()

        # Testing with all classifiers
        fs_pred = []
        for i in range(len(self.feature_sets)):
            fs_array = feature_values_array[:, self.feature_sets[i]]
            fs_pred.append(self.classifiers[i].predict(fs_array))
        fs_pred = numpy.array(fs_pred).transpose()

        # Calculating Uncertainty
        trust = []
        for i in range(len(feature_values_array)):
            trust.append(sum(fs_pred[i] == predicted_classes[i]) / len(fs_pred[i]))

        return np.asarray(trust)

    def uncertainty_calculator_name(self):
        return 'FeatureBagging Uncertainty (' + str(self.n_baggers) + '/' + str(self.bag_type) + ')'


class ReconstructionLoss(UncertaintyCalculator):
    """
    Defines a trust strategy that uses the reconstruction error of an autoencoder as uncertainty measure
    """

    def __init__(self, x_train, enc_tag='simple'):
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
        Returns the name of the strategy to calculate trust score (as string)
        :param main_folder: the folder where to save the details of the calculator
        :param tag: tag to name files
        """
        return {"enc_tag": self.enc_tag}

    def uncertainty_scores(self, feature_values_array, proba_array, classifier):
        """
        Returns the trust after executing a given amount of simulations around the feature values
        Score ranges from 0 (no agreement) to 1 (full agreement)

        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
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
