import copy
import os.path

import joblib
import numpy as np
import pandas
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from sprout.utils import general_utils
from sprout.utils.Classifier import LogisticReg
from sprout.UncertaintyCalculator import EntropyUncertainty, ConfidenceInterval, ExternalUncertainty, \
    CombinedUncertainty, MultiCombinedUncertainty, NeighborsUncertainty, MonteCarlo, FeatureBagging


class SPROUTObject:

    def __init__(self, models_folder):
        """
        Constructor for the SPROUT object
        """
        self.trust_calculators = []
        self.models_folder = models_folder

    def compute_data_trust(self, data_point, classifier, verbose=False, as_pandas=True):
        """
        Outputs an array / DataFrame containing trust measures for a data point
        :param data_point: test data point
        :param classifier: classifier to evaluate trust of
        :param verbose: False if you want to suppress debug information
        :param as_pandas: True if outputs has to e a Dataframe, False if ndarray
        :return:
        """
        item = data_point.reshape(-1, 1).transpose()
        return self.compute_set_trust(item, classifier, classifier.predict_prob(item), verbose, as_pandas)

    def compute_set_trust(self, data_set, classifier, y_proba=None, verbose=True, as_pandas=True):
        """
        Outputs an array / DataFrame containing trust measures for each data point in the dataset
        :param data_set: test dataset
        :param y_proba: probabilities calculated by the classifier to evaluate trust of
        :param classifier: classifier to evaluate trust of
        :param verbose: False if you want to suppress debug information
        :param as_pandas: True if outputs has to e a Dataframe, False if ndarray
        :return:
        """
        out_df = pd.DataFrame()
        if y_proba is None:
            y_proba = classifier.predict_proba(data_set)
        for calculator in self.trust_calculators:
            if verbose:
                print("Calculating Trust Strategy: " + calculator.strategy_name())
            start_ms = general_utils.current_ms()
            trust_scores = calculator.uncertainty_scores(data_set, y_proba, classifier)
            if type(trust_scores) is dict:
                for key in trust_scores:
                    out_df[calculator.strategy_name() + "_" + str(key)] = trust_scores[key]
            else:
                out_df[calculator.strategy_name()] = trust_scores
            if verbose:
                print("Completed in " + str(general_utils.current_ms() - start_ms) + " ms for " + str(len(data_set)) +
                      " items, " + str((general_utils.current_ms() - start_ms) / len(data_set)) + " ms per item")

        # Chooses output format
        if as_pandas:
            return out_df
        else:
            return out_df.to_numpy()

    def add_all_calculators(self, x_train, y_train, label_names, combined_clf, combined_clfs):
        """
        Adds all trust calculators to the SPROUT object
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param label_names: unique names in the label
        :param combined_clf: classifier used for CM4
        :param combined_clfs: classifier sets used for CM5
        """
        self.add_calculator_confidence(x_train=x_train, y_train=y_train)
        self.add_calculator_entropy(n_classes=len(label_names))
        self.add_calculator_external(classifier=LogisticReg(), x_train=x_train, y_train=y_train, n_classes=len(label_names))
        self.add_calculator_combined(classifier=combined_clf, x_train=x_train, y_train=y_train, n_classes=len(label_names))
        for cc in combined_clfs:
            self.add_calculator_multicombined(clf_set=cc, x_train=x_train, y_train=y_train, n_classes=len(label_names))
        self.add_calculator_neighbour(x_train=x_train, y_train=y_train, label_names=label_names)
        self.add_calculator_montecarlo(x_train=x_train, y_train=y_train)
        self.add_calculator_featurebagging(x_train=x_train, y_train=y_train, n_baggers=50)

    def add_calculator_confidence(self, x_train, y_train, confidence_level=0.9999):
        """
        Confidence Interval Calculator (CM1 from paper)
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param confidence_level: size of the confidence interval (default: 0.9999)
        """
        self.trust_calculators.append(
                ConfidenceInterval(x_train=(x_train if isinstance(x_train, np.ndarray) else x_train.to_numpy()),
                                   y_train=y_train,
                                   confidence_level=confidence_level))

    def add_calculator_entropy(self, n_classes):
        """
        Entropy Calculator (CM2 from paper)
        :param n_classes: number of classes of the label
        """
        self.trust_calculators.append(EntropyUncertainty(norm=n_classes))

    def add_calculator_bayes(self, x_train, y_train, n_classes):
        """
        External Trust Calculator using Bayes (CM3 in the paper)
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param n_classes: number of classes of the label
        """
        self.add_calculator_external(classifier=GaussianNB(), x_train=x_train, y_train=y_train, n_classes=n_classes)

    def add_calculator_external(self, classifier, x_train, y_train, n_classes):
        """
        External Trust Calculator (CM3 in the paper)
        :param classifier: classifier to be used as del_clf
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param n_classes: number of classes of the label
        """
        self.trust_calculators.append(
            ExternalUncertainty(del_clf=classifier, x_train=x_train, y_train=y_train, norm=n_classes))

    def add_calculator_combined(self, classifier, x_train, y_train, n_classes):
        """
        Combined Trust Calculator (CM4 in the paper)
        :param classifier: classifier to be used as del_clf
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param n_classes: number of classes of the label
        """
        self.trust_calculators.append(
            CombinedUncertainty(del_clf=classifier, x_train=x_train, y_train=y_train, norm=n_classes))

    def add_calculator_multicombined(self, clf_set, x_train, y_train, n_classes):
        """
        Multi-Combined Trust Calculator (CM5 in the paper)
        :param clf_set: classifiers (array) to be used as del_clf
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param n_classes: number of classes of the label
        """
        self.trust_calculators.append(
            MultiCombinedUncertainty(clf_set=clf_set, x_train=x_train, y_train=y_train, norm=n_classes))

    def add_calculator_neighbour(self, x_train, y_train, label_names, k=19):
        """
        Neighbour-based Trust Calculator (CM6 in the paper)
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param label_names: unique names in the label
        :param k: k parameter for kNN search
        """
        self.trust_calculators.append(NeighborsUncertainty(x_train=x_train, y_train=y_train, k=k, labels=label_names))

    def add_calculator_montecarlo(self, x_train, y_train):
        self.trust_calculators.append(MonteCarlo(x_train, y_train))

    def add_calculator_featurebagging(self, x_train, y_train, n_baggers=50):
        self.trust_calculators.append(FeatureBagging(x_train, y_train, n_baggers))

    def predict_misclassifications(self, model_tag, trust_set):
        clf = self.load_model(model_tag)
        sp_df = copy.deepcopy(trust_set)
        if clf is not None:
            if isinstance(trust_set, pandas.DataFrame):
                x_test = sp_df.to_numpy()
            else:
                x_test = sp_df
            predictions = clf.predict(x_test)
            sp_df["pred"] = predictions
        else:
            print("Unable to load model with tag '" + str(model_tag) + "'")

        return sp_df, clf

    def load_model(self, model_tag):
        clf = None
        if os.path.exists(self.models_folder):
            model_file = self.models_folder + str(model_tag) + ".joblib"
            clf = joblib.load(model_file)
        else:
            print("Models folder '" + self.models_folder + "' does not exist")

        return clf

