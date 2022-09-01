import random

import numpy
import pandas
import pandas as pd
import scipy.stats

import numpy as np

from sklearn.neighbors import NearestNeighbors
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from sprout.utils.sprout_utils import get_classifier_name
from sprout.utils.general_utils import current_ms


class UncertaintyCalculator:
    """
    Abstract Class for trust calculators. Methods to be overridden are trust_strategy_name and trust_scores
    """

    def trust_strategy_name(self):
        """
        Returns the name of the strategy to calculate trust score (as string)
        """
        pass

    def trust_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        pass


class EntropyUncertainty(UncertaintyCalculator):
    """
    Computes Trust via Entropy of the probability array for a given data point.
    Higher entropy means low trust / confidence
    """

    def __init__(self, norm):
        """
        Constructor Method
        :param norm: number of classes for normalization process
        """
        norm_array = np.full(norm, 1 / norm)
        self.normalization = (-norm_array * np.log2(norm_array)).sum()
        return

    def trust_score(self, feature_values, proba, classifier):
        """
        Returns the entropy for a given prediction array
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point
        :param classifier: the classifier used for classification
        :return: entropy score in the range [0, 1]
        """

        val = np.delete(proba, np.where(proba == 0))
        p = val / val.sum()
        entropy = (-p * np.log2(p)).sum()
        return (self.normalization - entropy) / self.normalization

    def trust_scores(self, feature_values_array, proba_array, classifier):
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
                trust.append(self.trust_score(feature_values_array[i], proba_array[i], classifier))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(trust)

    def trust_strategy_name(self):
        return 'Entropy Calculator'


class NeighborsUncertainty(UncertaintyCalculator):
    """
    Computes Trust via Agreement with label predictions of neighbours.
    Reports both on the trust and on the details for the neighbours.
    """

    def __init__(self, x_train, y_train, k, labels):
        self.x_train = x_train
        self.y_train = y_train
        self.n_neighbors = k
        self.labels = labels

    def trust_strategy_name(self):
        return 'Trust Calculator on ' + str(self.n_neighbors) + ' Neighbors'

    def trust_scores(self, feature_values, proba, classifier):
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


class ExternalUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that runs an external classifer and calculates its confidence in the result
    """

    def __init__(self, del_clf, x_train, y_train, norm):
        self.del_clf = del_clf
        self.del_clf.fit(x_train, y_train)
        self.trust_measure = EntropyUncertainty(norm)
        print("[ExternalTrust] Fitting of '" + get_classifier_name(del_clf) + "' Completed")

    def trust_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        return self.trust_measure.trust_scores(feature_values_array,
                                               self.del_clf.predict_proba(feature_values_array),
                                               self.del_clf)

    def trust_strategy_name(self):
        return 'External Calculator (' + get_classifier_name(self.del_clf) + ')'


class CombinedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, del_clf, x_train, y_train, norm):
        self.del_clf = del_clf
        start_time = current_ms()
        self.del_clf.fit(x_train, y_train)
        print("[CombinedTrust] Fitting of '" + get_classifier_name(del_clf) + "' Completed in " +
              str(current_ms() - start_time) + " ms")
        self.trust_measure = EntropyUncertainty(norm)

    def trust_scores(self, feature_values_array, proba_array, classifier):
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
        entropy = self.trust_measure.trust_scores(feature_values_array, proba_array, classifier)
        other_entropy = self.trust_measure.trust_scores(feature_values_array,
                                                        self.del_clf.predict_proba(feature_values_array),
                                                        self.del_clf)
        return np.where(pred == other_pred, (entropy + other_entropy) / 2, -(entropy + other_entropy) / 2)

    def trust_strategy_name(self):
        return 'Combined Calculator (' + get_classifier_name(self.del_clf) + ')'


class MultiCombinedUncertainty(UncertaintyCalculator):
    """
    Defines a trust strategy that uses another classifer and calculates a combined confidence
    It uses the main classifier plus the additional classifier to calculate an unified confidence score
    """

    def __init__(self, clf_set, x_train, y_train, norm):
        self.trust_set = []
        self.tag = ""
        start_time = current_ms()
        for clf in clf_set:
            self.trust_set.append(CombinedUncertainty(clf, x_train, y_train, norm))
            self.tag = self.tag + get_classifier_name(clf)[0] + get_classifier_name(clf)[-1]
        self.tag = str(len(self.trust_set)) + " - " + self.tag
        print("[MultiCombinedTrust] Fitting of " + str(len(clf_set)) + " classifiers completed in "
              + str(current_ms() - start_time) + " ms")

    def trust_scores(self, feature_values_array, proba_array, classifier):
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
            multi_trust = multi_trust + combined_trust.trust_scores(feature_values_array, proba_array, classifier)
        return multi_trust / len(self.trust_set)

    def trust_strategy_name(self):
        return 'Multiple Combined Calculator (' + str(self.tag) + ' classifiers)'


class ConfidenceInterval(UncertaintyCalculator):
    """
    Defines a trust strategy that calculates confidence intervals to derive trust
    """

    def __init__(self, x_train, y_train, confidence_level=0.9999):
        self.confidence_level = confidence_level
        self.intervals = {}
        self.labels = numpy.unique(y_train)
        for label in self.labels:
            self.intervals[label] = []
            data = x_train[y_train == label, :]
            for i in range(0, len(x_train[0])):
                feature = data[:, i]
                self.intervals[label].append(scipy.stats.t.interval(confidence_level,
                                                                    len(feature) - 1,
                                                                    loc=np.mean(feature),
                                                                    scale=scipy.stats.sem(feature)))

    def trust_score(self, feature_values, proba, classifier):
        """
        Returns the degree to which a data point complies with a confidence interval. Agnostic of the classifier
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point (UNUSED)
        :param classifier: the classifier used for classification (UNUSED)
        :return: trust score using confidence intervals
        """
        int_trust = 0
        predicted_label = numpy.argmax(proba)
        for i in range(0, len(feature_values)):
            if (np.isfinite(self.intervals[predicted_label][i][0])) & \
                    (np.isfinite(self.intervals[predicted_label][i][1])):
                if (feature_values[i] < self.intervals[predicted_label][i][0]) | \
                        (feature_values[i] > self.intervals[predicted_label][i][1]):
                    int_trust = int_trust + 1
        return int_trust / len(feature_values)

    def trust_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param classifier: the classifier used for classification
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        trust = []
        if isinstance(feature_values_array, pandas.DataFrame):
            feature_values_array = feature_values_array.to_numpy()
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                trust.append(self.trust_score(feature_values_array[i], proba_array[i], classifier))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return np.asarray(trust)

    def trust_strategy_name(self):
        return 'Confidence Interval (' + str(self.confidence_level) + '%)'


class MonteCarlo(UncertaintyCalculator):
    """
    Defines a trust strategy that uses a Monte Carlo simulation for each class
    """

    def __init__(self, x_train, y_train, mc_iterations=10):
        self.mc_iterations = mc_iterations
        self.labels = numpy.unique(y_train)
        self.averages = {}
        self.stds = {}

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        start_time = current_ms()
        for label in self.labels:
            data = x_train[numpy.where(y_train == label)[0], :]
            self.averages[label] = sum(data) / len(data)
            self.stds[label] = numpy.std(data, axis=0)

        print("MonteCarlo initialized in " + str(current_ms() - start_time) + " ms")

    def trust_scores(self, feature_values_array, proba_array, classifier):
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

        # Generating MC Simulations for inputs
        mc_x = []
        for i in range(len(feature_values_array)):
            features = feature_values_array[i]
            for _ in range(self.mc_iterations):
                mc_x.append([random.gauss(m, s)
                             for m, s in zip(features, self.stds[predicted_classes[i]])])
        mc_x = np.array(mc_x)

        # Calculating predictions
        mc_predict = classifier.predict_proba(mc_x)

        # Calculating Uncertainty
        trust = []
        for i in range(len(feature_values_array)):
            mc_probas = mc_predict[i * self.mc_iterations:(i + 1) * self.mc_iterations, :]
            mc_avg = sum(mc_probas) / len(mc_probas)
            score = 1 - numpy.average(numpy.std(mc_probas, axis=0))
            if numpy.argmax(mc_avg) != predicted_classes[i]:
                score = -score
            trust.append(score)

        return np.asarray(trust)

    def trust_strategy_name(self):
        return 'MonteCarlo Uncertainty'


class FeatureBagging(UncertaintyCalculator):
    """
    Defines a trust strategy that uses a Monte Carlo simulation for each class
    """

    def __init__(self, x_train, y_train, n_baggers=10):
        self.feature_sets = []
        self.classifiers = []
        if isinstance(n_baggers, int):
            self.n_baggers = n_baggers
        else:
            self.n_baggers = 10

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
        bag_features = int(n_features*bag_rate)

        for i in tqdm(range(self.n_baggers), "Building Feature Baggers"):
            fs = random.sample(range(n_features), bag_features)
            fs.sort()
            self.feature_sets.append(fs)
            classifier = DecisionTreeClassifier()
            classifier.fit(x_train[:, fs], y_train)
            self.classifiers.append(classifier)

    def trust_scores(self, feature_values_array, proba_array, classifier):
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
            trust.append(sum(fs_pred[i] == predicted_classes[i])/len(fs_pred[i]))

        return np.asarray(trust)

    def trust_strategy_name(self):
        return 'FeatureBagging Uncertainty (' + str(self.n_baggers) + ")"
