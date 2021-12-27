import lime
import lime.lime_tabular

import numpy as np


class TrustCalculator:

    def trust_strategy_name(self):
        """
        Returns the name of the strategy to calculate trust score (as string)
        """
        pass

    def trust_score(self, feature_values, proba):
        """
        Method to compute trust score for a single data point
        :param feature_values: the feature values of the data point
        :param proba: the probability array assigned by the algorithm to the data point
        """
        pass

    def trust_scores(self, feature_values_array, proba_array):
        """
        Method to compute trust score for a set of data points
        :param feature_values_array: the feature values of the data points in the test set
        :param proba_array: the probability arrays assigned by the algorithm to the data points
        :return: array of trust scores
        """
        trust = []
        if len(feature_values_array) == len(proba_array):
            for i in range(0, len(proba_array)):
                # print(i)
                trust.append(self.trust_score(feature_values_array[i], proba_array[i]))
        else:
            print("Items of the feature set have a different cardinality wrt probabilities")
        return trust


class LimeTrust(TrustCalculator):

    def __init__(self, x_data, y_data, column_names, class_names, classifier):
        self.model = classifier
        self.column_names = column_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=x_data,
                                                                training_labels=y_data,
                                                                feature_names=column_names,
                                                                class_names=class_names,
                                                                verbose=True)

    def trust_score(self, feature_values, proba):
        val_exp = self.explainer.explain_instance(data_row=feature_values,
                                                  predict_fn=self.model.predict_proba,
                                                  num_features=len(self.column_names))
        return {"LIME_Sum": sum(x[1] for x in val_exp.local_exp[1]),
                "LIME_Intercept": val_exp.intercept,
                "LIME_Pred": val_exp.local_pred}

    def trust_strategy_name(self):
        return 'Lime Trust Calculator'


class EntropyTrust(TrustCalculator):

    def __init__(self):
        return

    def trust_score(self, feature_values, proba):
        p = proba / proba.sum()
        return (-p*np.log2(p)).sum()

    def trust_strategy_name(self):
        return 'Entropy Calculator'


