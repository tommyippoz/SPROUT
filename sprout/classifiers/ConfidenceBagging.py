import copy
import random

import numpy
from sklearn.utils.multiclass import unique_labels

from sprout.classifiers.Classifier import Classifier


class ConfidenceBagging(Classifier):
    """
    Class for creating Unsupervised boosting ensembles
    """

    def __init__(self, clf, n_base: int = 10, max_features: float = 0.7,
                 sampling_ratio: float = 0.7, perc_decisors: float = None, n_decisors: int = None):
        super().__init__(clf)
        if n_base > 1:
            self.n_base = n_base
        else:
            print("Ensembles have to be at least 2")
            self.n_base = 10
        if max_features is not None and 0 < max_features <= 1:
            self.max_features = max_features
        else:
            self.max_features = 0.7
        if sampling_ratio is not None and 0 < sampling_ratio <= 1:
            self.sampling_ratio = sampling_ratio
        else:
            self.sampling_ratio = 0.7
        if perc_decisors is not None and 0 < perc_decisors <= 1:
            if n_decisors is not None and 0 < n_decisors <= self.n_base:
                print('Both perc_decisors and n_decisors are specified, prioritizing perc_decisors')
            self.n_decisors = int(self.n_base*perc_decisors) if int(self.n_base*perc_decisors) > 0 else 1
        elif n_decisors is not None and 0 < n_decisors <= self.n_base:
            self.n_decisors = n_decisors
        else:
            self.n_decisors = 1 + int(self.n_base / 2)
        self.base_learners = []
        self.feature_sets = []

    def fit(self, X, y=None):
        train_n = len(X)
        self.classes_ = unique_labels(y) if y is not None else [0, 1]
        bag_features_n = int(X.shape[1]*self.max_features)
        samples_n = int(train_n * self.sampling_ratio)
        for learner_index in range(0, self.n_base):
            # Draw samples
            features = random.sample(range(X.shape[1]), bag_features_n)
            features.sort()
            self.feature_sets.append(features)
            sample_x, sample_y = self.draw_samples(X, y, samples_n)
            sample_x = sample_x[:, features]
            if len(features) == 1:
                sample_x = sample_x.reshape(-1, 1)
            # Train learner
            learner = copy.deepcopy(self.clf)
            learner.fit(sample_x, sample_y)
            # Test Learner
            self.base_learners.append(learner)

        # Compliance with SKLEARN and PYOD

        self.X_ = X
        self.y_ = y
        self.feature_importances_ = self.compute_feature_importances()

    def draw_samples(self, X, y, samples_n):
        indexes = numpy.random.choice(X.shape[0], samples_n, replace=False, p=None)
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

    def predict_proba(self, X):
        # Scoring probabilities, ends with a
        proba_array = []
        conf_array = []
        for i in range(0, self.n_base):
            predictions = self.base_learners[i].predict_proba(X[:, self.feature_sets[i]])
            proba_array.append(predictions)
            conf_array.append(numpy.max(predictions, axis=1))
        # 3d matrix (clf, row, probability for class)
        proba_array = numpy.asarray(proba_array)
        # 2dim matrix (clf, confidence for row)
        conf_array = numpy.asarray(conf_array).transpose()

        # Choosing the most confident self.n_decisors to compute final probabilities
        proba = numpy.zeros(proba_array[0].shape)
        all_conf = -numpy.sort(-conf_array, axis=1)
        conf_thrs = all_conf[:, self.n_decisors-1]
        for i in range(0, X.shape[0]):
            proba[i] = numpy.average(proba_array[numpy.where(conf_array[i] >= conf_thrs[i]), i, :], axis=1)

        # Final averaged Result
        return proba

    def classifier_name(self):
        clf_name = self.clf.classifier_name() if isinstance(self.clf, Classifier) else self.clf.__class__.__name__
        if clf_name == 'Pipeline':
            keys = list(self.clf.named_steps.keys())
            clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
        return "ConfidenceBagger(" + str(clf_name) + "-" + \
               str(self.n_base) + "-" + str(self.n_decisors) + "-" + \
               str(self.max_features) + "-" + str(self.sampling_ratio) + ")"


class ConfidenceBaggingWeighted(ConfidenceBagging):

    def __init__(self, clf, n_base: int = 10, max_features: float = 0.7, sampling_ratio: float = 0.7):
        super().__init__(clf, n_base, max_features, sampling_ratio, None, None)

    def predict_proba(self, X):
        # Scoring probabilities, ends with a
        proba_array = []
        conf_array = []
        for i in range(0, self.n_base):
            predictions = self.base_learners[i].predict_proba(X[:, self.feature_sets[i]])
            proba_array.append(predictions)
            conf_array.append(numpy.max(predictions, axis=1))
        # 3d matrix (clf, row, probability for class)
        proba_array = numpy.asarray(proba_array)
        # 2dim matrix (clf, confidence for row)
        conf_array = numpy.asarray(conf_array)

        # Weihting probas using confidence
        proba = numpy.zeros(proba_array[0].shape)
        for i in range(0, X.shape[0]):
            proba[i] = numpy.sum(proba_array[:, i, :].T*conf_array[:, i], axis=1)/numpy.sum(conf_array[:, i])

        # Final averaged Result
        return proba

    def classifier_name(self):
        clf_name = self.clf.classifier_name() if isinstance(self.clf, Classifier) else self.clf.__class__.__name__
        if clf_name == 'Pipeline':
            keys = list(self.clf.named_steps.keys())
            clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
        return "ConfidenceBaggerWeighted(" + str(clf_name) + "-" + \
               str(self.n_base) + "-" + str(self.max_features) + "-" + str(self.sampling_ratio) + ")"

