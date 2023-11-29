import copy

import numpy
from sklearn.utils.multiclass import unique_labels

from sprout.classifiers.Classifier import Classifier


class ConfidenceBoosting(Classifier):
    """
    Class for creating Confidence Boosting ensembles
    """

    def __init__(self, clf, n_base: int = 10, learning_rate: float = None,
                 sampling_ratio: float = 0.5, contamination: float = None, conf_thr: float = 0.8):
        """
        Constructor
        :param clf: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param learning_rate: learning rate for updating dataset weights
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param contamination: percentage of anomalies. TRThis is used to automatically devise conf_thr
        :param conf_thr: threshold of acceptance for confidence scores. Lower confidence means untrustable result
        """
        super().__init__(clf)
        self.proba_thr = None
        self.conf_thr = conf_thr
        self.contamination = contamination
        if n_base > 1:
            self.n_base = n_base
        else:
            print("Ensembles have to be at least 2")
            self.n_base = 10
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 2
        if sampling_ratio is not None:
            self.sampling_ratio = sampling_ratio
        else:
            self.sampling_ratio = 1 / n_base ** (1 / 2)
        self.base_learners = []

    def fit(self, X, y=None):
        """
        Training function for the confidence boosting ensemble
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        train_n = len(X)
        self.classes_ = unique_labels(y) if y is not None else [0, 1]
        samples_n = int(train_n * self.sampling_ratio)
        weights = numpy.full(train_n, 1 / train_n)
        for learner_index in range(0, self.n_base):
            # Draw samples
            sample_x, sample_y = self.draw_samples(X, y, weights, samples_n)
            # Train learner
            learner = copy.deepcopy(self.clf)
            learner.fit(sample_x, sample_y)
            # Test Learner
            y_proba = learner.predict_proba(X)
            y_conf = numpy.max(y_proba, axis=1)
            p_thr = self.define_conf_thr(target=self.conf_thr, confs=y_conf)
            self.base_learners.append(learner)
            # Update Weights
            update_flag = numpy.where(y_conf >= p_thr, 0, 1)
            weights = weights * (1 + self.learning_rate * update_flag)
            weights = weights / sum(weights)
        self.proba_thr = self.define_proba_thr(target=self.contamination,
                                               probs=self.predict_proba(X)) \
            if self.contamination is not None else 0.5

        # Compliance with SKLEARN and PYOD
        self.X_ = X
        self.y_ = y
        self.feature_importances_ = self.compute_feature_importances()

    def draw_samples(self, X, y, weights, samples_n):
        indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
        sample_x = numpy.asarray(X[indexes, :])
        # If data is labeled we also have to refine labels
        if y is not None and hasattr(self, 'classes_') and self.classes_ is not None and len(self.classes_) > 1:
            sample_y = y[indexes]
            sample_labels = unique_labels(sample_y)
            missing_labels = [item for item in self.classes_ if item not in sample_labels]
            # And make sure that there is at least a sample for each class of the problem
            if missing_labels is not None and len(missing_labels) > 0:
                # For each missing class
                for missing_class in missing_labels:
                    miss_class_indexes = numpy.asarray(numpy.where(y == missing_class)[0])
                    new_sampled_index = numpy.random.choice(miss_class_indexes, None, replace=False, p=None)
                    X_missing_class = X[new_sampled_index, :]
                    sample_x = numpy.append(sample_x, [X_missing_class], axis=0)
                    sample_y = numpy.append(sample_y, missing_class)
        else:
            sample_y = None
        return sample_x, sample_y

    def define_conf_thr(self, confs, target=None, delta=0.01):
        target_thr = target
        left_bound = min(confs)
        right_bound = max(confs)
        c_thr = (right_bound + left_bound) / 2
        a = numpy.average(confs < 0.6)
        b = numpy.average(confs < 0.9)
        actual_thr = numpy.average(confs < c_thr)
        while abs(actual_thr - target_thr) > delta and abs(right_bound - left_bound) > 0.001:
            if actual_thr < target_thr:
                left_bound = c_thr
                c_thr = (c_thr + right_bound) / 2
            else:
                right_bound = c_thr
                c_thr = (c_thr + left_bound) / 2
            actual_thr = numpy.average(confs < c_thr)
        return c_thr

    def define_proba_thr(self, probs, target=None, delta=0.01):
        """
        Method for finding a confidence threshold based on the expected contamination (iterative)
        :param probs: probabilities to find threshold of
        :param target: the quantity to be used as reference for gettng to the threshold
        :param delta: the tolerance to stop recursion
        :return: a float value to be used as threshold for updating weights in boosting
        """
        target_cont = target
        p_thr = 0.5
        left_bound = 0.5
        right_bound = 1
        actual_cont = numpy.average(probs[:, 0] < p_thr)
        while abs(actual_cont - target_cont) > delta and abs(right_bound - left_bound) > 0.01:
            if actual_cont < target_cont:
                left_bound = p_thr
                p_thr = (p_thr + right_bound) / 2
            else:
                right_bound = p_thr
                p_thr = (p_thr + left_bound) / 2
            actual_cont = numpy.average(probs[:, 0] < p_thr)
        return p_thr

    def predict_proba(self, X):
        """
        Method to compute prediction probabilities (i.e., normalized logits) of a classifier
        :param X: the test set
        :return: array of probabilities for each data point and each class
        """
        proba = numpy.zeros((X.shape[0], len(self.classes_)))
        for clf in self.base_learners:
            predictions = clf.predict_proba(X)
            proba += predictions
        return proba / self.n_base

    def predict_confidence(self, X):
        """
        Method to compute the confidence in predictions of a classifier
        :param X: the test set
        :return: array of confidence scores
        """
        conf = numpy.zeros(X.shape[0])
        for clf in self.base_learners:
            if hasattr(clf, 'predict_confidence') and callable(clf.predict_confidence):
                c_conf = clf.predict_confidence(X)
            else:
                y_proba = clf.predict_proba(X)
                c_conf = numpy.max(y_proba, axis=1)
            conf += c_conf
        return conf / self.n_base

    def get_feature_importances(self):
        """
        Placeholder, to be implemented if possible
        :return: feature importances (to be tested)
        """
        fi = []
        for clf in self.base_learners:
            c_fi = clf.get_feature_importances()
            fi.append(c_fi)
        fi = numpy.asarray(fi)
        return numpy.average(fi, axis=1)

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :param X: the test set
        :return: array of predicted class
        """
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[1 * (proba[:, 0] < self.proba_thr)]
        else:
            return self.classes_[numpy.argmax(proba, axis=1)]

    def classifier_name(self):
        clf_name = self.clf.classifier_name() if isinstance(self.clf, Classifier) else self.clf.__class__.__name__
        if clf_name == 'Pipeline':
            keys = list(self.clf.named_steps.keys())
            clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
        return "ConfidenceBooster(" + str(clf_name) + "-" + \
               str(self.n_base) + "-" + str(self.conf_thr) + "-" + \
               str(self.learning_rate) + "-" + str(self.sampling_ratio) + ")"


class ConfidenceBoostingWeighted(ConfidenceBoosting):
    """
    Class for creating Confidence Boosting ensembles with weighted predictions
    """

    def __init__(self, clf, n_base: int = 10, learning_rate: float = None,
                 sampling_ratio: float = 0.5, contamination: float = None, conf_thr: float = 0.8):
        """
        Constructor
        :param clf: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param learning_rate: learning rate for updating dataset weights
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param contamination: percentage of anomalies. TRThis is used to automatically devise conf_thr
        :param conf_thr: threshold of acceptance for confidence scores. Lower confidence means untrustable result
        """
        super().__init__(clf, n_base, learning_rate, sampling_ratio, contamination, conf_thr)

    def predict_proba(self, X):
        """
        Method to compute prediction probabilities (i.e., normalized logits) of a classifier
        :param X: the test set
        :return: array of probabilities for each data point and each class
        """
        proba_array = []
        conf_array = []
        for clf in self.base_learners:
            predictions = clf.predict_proba(X)
            proba_array.append(predictions)
            conf_array.append(numpy.max(predictions, axis=1))
            # 3d matrix (clf, row, probability for class)
        proba_array = numpy.asarray(proba_array)
        # 2dim matrix (clf, confidence for row)
        conf_array = numpy.asarray(conf_array)

        # Weighting probas using confidence
        proba = numpy.zeros(proba_array[0].shape)
        for i in range(0, X.shape[0]):
            proba[i] = numpy.sum(proba_array[:, i, :].T * conf_array[:, i], axis=1) / numpy.sum(conf_array[:, i])

        # Final averaged Result
        return proba

    def classifier_name(self):
        tag = super().classifier_name()
        return tag.replace("ConfidenceBooster", "ConfidenceBoosterWeighted")
