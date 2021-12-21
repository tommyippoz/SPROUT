import lime
import lime.lime_tabular

import numpy as np


class TrustCalculator:

    def trust_strategy_name(self):
        pass

    def trust_score(self, value):
        pass

    def trust_scores(self, values):
        trust = []
        for value in values:
            trust.append(self.trust_score(value))
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

    def trust_score(self, value):
        val_exp = self.explainer.explain_instance(data_row=value,
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

    def trust_score(self, value):
        print()
        p = value / value.sum()
        return (-p*np.log2(p)).sum()

    def trust_strategy_name(self):
        return 'Entropy Calculator'
