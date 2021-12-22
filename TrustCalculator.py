import lime
import lime.lime_tabular

import numpy as np


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

    def __init__(self, x_data, y_data, column_names, class_names):
        self.column_names = column_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=x_data,
                                                                training_labels=y_data,
                                                                feature_names=column_names,
                                                                class_names=class_names,
                                                                verbose=True)

    def trust_score(self, feature_values, proba, classifier):
        val_exp = self.explainer.explain_instance(data_row=feature_values,
                                                  predict_fn=classifier.predict_proba,
                                                  num_features=len(self.column_names))
        return {"LIME_Sum": sum(x[1] for x in val_exp.local_exp[1]),
                "LIME_Intercept": val_exp.intercept,
                "LIME_Pred": val_exp.local_pred}

    def trust_strategy_name(self):
        return 'Lime Trust Calculator'


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
