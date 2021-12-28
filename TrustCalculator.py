import lime
import lime.lime_tabular
import shap
import numpy as np
from sklearn.neighbors import NearestNeighbors

class TrustCalculator:
    """
    Abstract Class for trust calculators. Methods to be overridden are trust_strategy_name and trust_score
    """

    def trust_strategy_name(self):
        """
        Returns the name of the strategy to calculate trust score (as string)
        """
        pass

    def trust_score(self, feature_values, proba, classifier):
        """
        Method to compute trust score for a single data point
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point
        :param classifier: the classifier used for classification
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
        trust = []
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                trust.append(self.trust_score(feature_values_array[i], proba_array[i], classifier))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return trust


class LimeTrust(TrustCalculator):
    """
    Computes Trust via LIME Framework for explainability.
    Reports on 3 different trust metrics: Sum, Intercept, Pred
    """

    def __init__(self, x_data, y_data, column_names, class_names, max_samples):
        self.max_samples = max_samples
        self.column_names = column_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=x_data,
                                                                training_labels=y_data,
                                                                feature_names=column_names,
                                                                class_names=class_names,
                                                                verbose=True)

    def trust_score(self, feature_values, proba, classifier):
        """
        Outputs an array of three items for each data point, containing Sum, Intercept, Pred
        :param feature_values:
        :param proba:
        :param classifier:
        :return:
        """
        val_exp = self.explainer.explain_instance(data_row=feature_values,
                                                  predict_fn=classifier.predict_prob,
                                                  num_features=len(self.column_names),
                                                  num_samples=self.max_samples)
        return {"Sum": sum(x[1] for x in val_exp.local_exp[1]),
                "Intercept": val_exp.intercept,
                "Pred": val_exp.local_pred}

    def trust_scores(self, feature_values_array, proba_array, classifier):
        """
        Method to compute trust score for a set of data points
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :param classifier: the classifier used for classification
        :return: array of trust scores
        """
        trust_sum = []
        trust_int = []
        trust_pred = []
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                lime_out = self.trust_score(feature_values_array[i], proba_array[i], classifier)
                trust_sum.append(lime_out["Sum"])
                trust_int.append(lime_out["Intercept"][1])
                trust_pred.append(lime_out["Pred"][0])
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return {"Sum": trust_sum, "Intercept": trust_int, "Pred": trust_pred}

    def trust_strategy_name(self):
        return 'LIME Trust Calculator (' + str(self.max_samples) + ')'


class EntropyTrust(TrustCalculator):
    """
    Computes Trust via Entropy of the probability array for a given data point.
    Higher entropy means low trust / confidence
    """

    def __init__(self):
        return

    def trust_score(self, feature_values, proba, classifier):
        """
        Returns the entropy for a given prediction array
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point
        :param classifier: the classifier used for classification
        :return: entropy score in the range [0, 1]
        """
        p = proba / proba.sum()
        if 0 in p:
            return 0
        else:
            return (-p*np.log2(p)).sum()

    def trust_strategy_name(self):
        return 'Entropy Calculator'


class NativeTrust(TrustCalculator):
    """
    Calls the existing function (if any) to calculate confidence in a prediction for a given data point.
    Only works with unsupervised algorithms belonging to the library PYOD.
    In any other case it returns -1.
    """

    def __init__(self):
        return

    def trust_score(self, feature_values, proba, classifier):
        """
        Returns the native confidence in a given prediction, or -1 if this native confidence is not available
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point
        :param classifier: the classifier used for classification
        :return: native confidence score
        """
        try:
            return classifier.predict_confidence(feature_values)
        except:
            return -1

    def trust_strategy_name(self):
        return 'Native Trust Calculator'


class SHAPTrust(TrustCalculator):
    """
    Computes Trust via SHAP Framework for explainability.
    Reports on 3 different trust metrics: Sum, Intercept, Pred
    """

    def __init__(self, x_data, max_samples):
        self.x_data = x_data
        self.max_samples = max_samples

    def trust_scores(self, feature_values_array, proba_array, classifier):
        """
        TO BE DEBUGGED. SOMETIMES IT CRASHES AND I DONT KNOW WHY
        :param feature_values_array:
        :param proba_array:
        :param classifier:
        :return:
        """
        explainer = shap.KernelExplainer(classifier.predict_prob,
                                         shap.sample(self.x_data, self.max_samples),
                                         link="logit")
        shap_values = explainer.shap_values(feature_values_array, nsamples=100, l1_reg="bic")
        return shap_values[0].sum(axis=1)

    def trust_score(self, feature_values, proba, classifier):
        """
        Not defined. Use trust_scores instead.
        :param feature_values:
        :param proba:
        :param classifier:
        :return:
        """
        pass

    def trust_strategy_name(self):
        return 'SHAP Trust Calculator (' + str(self.max_samples) + ')'

class NeighborsTrust(TrustCalculator):

    def __init__(self, X_train, X_test, y_train, classifier):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.classifier = classifier

    def find_neighbors(x_test, neighbors):
        near_neighbors = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(x_test)
        distances, indices = near_neighbors.kneighbors(x_test)
        return indices

    def neighbours_train(neighbors, X_train):
        X_train_neighbors = np.empty((0, len(X_train[0])), float)
        for n in range(len(neighbors)):
            X_train_neighbors = np.append(X_train_neighbors, [X_train[neighbors[n]]], axis=0)
        return X_train_neighbors

    def trust_score(self):
        n_neighbors = int(np.round(len(self.X_train) / len(self.X_test), 0) * len(self.X_train[0]) + 1)
        for i in range(len(self.X_test)):
            find_n = np.vstack([self.X_train, self.X_test[i]])
            neighbors = self.find_neighbors(find_n, n_neighbors)[len(self.find_neighbors(find_n, n_neighbors)) - 1]
            neighbors = neighbors[1:]  # find indexes neighbors
            # train_neighbors = self.neighbours_train(neighbors, self.X_train)  # train neighbors
            # array_class = self.classifier.fit(train_neighbors).predict_class()
            # for i in range(len(self.classifier)):
            # array_class = self.classifier[i].predict_class()
            # for j in range(len(array_class)):
                # array_knn[j] = array_knn[j] + [array_class[j]]
        # for i in range(len(self.classifier)):
            # trust_neighbors = Counter(array_knn[i][0]).most_common(1)
        # return trust_neighbors
