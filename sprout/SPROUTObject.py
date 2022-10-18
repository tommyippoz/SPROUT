import copy
import fnmatch
import os.path

import joblib
import pandas
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.naive_bayes import GaussianNB

from sprout.utils import general_utils
from sprout.UncertaintyCalculator import EntropyUncertainty, ConfidenceInterval, ExternalSupervisedUncertainty, \
    CombinedUncertainty, MultiCombinedUncertainty, NeighborsUncertainty, ProximityUncertainty, FeatureBagging, \
    ReconstructionLoss, \
    ExternalUnsupervisedUncertainty, MaxProbUncertainty, AgreementUncertainty
from sprout.utils.general_utils import get_full_class_name
from sprout.utils.sprout_utils import get_classifier_name, read_calculators


class SPROUTObject:

    def __init__(self, models_folder):
        """
        Constructor for the SPROUT object
        """
        self.trust_calculators = []
        self.models_folder = models_folder
        self.binary_adjudicator = None

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
                print("Calculating Trust Strategy: " + calculator.uncertainty_calculator_name())
            start_ms = general_utils.current_ms()
            if isinstance(data_set, pandas.DataFrame):
                data_set = data_set.to_numpy()
            trust_scores = calculator.uncertainty_scores(data_set, y_proba, classifier)
            if type(trust_scores) is dict:
                for key in trust_scores:
                    out_df[calculator.uncertainty_calculator_name() + "_" + str(key)] = trust_scores[key]
            else:
                out_df[calculator.uncertainty_calculator_name()] = trust_scores
            if verbose:
                print("Completed in " + str(general_utils.current_ms() - start_ms) + " ms for " + str(len(data_set)) +
                      " items, " + str((general_utils.current_ms() - start_ms) / len(data_set)) + " ms per item")

        # Chooses output format
        if as_pandas:
            return out_df
        else:
            return out_df.to_numpy()

    def add_calculator_confidence(self, x_train, y_train=None, confidence_level=0.9999):
        """
        Confidence Interval Calculator (CM1 from paper)
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param confidence_level: size of the confidence interval (default: 0.9999)
        """
        self.trust_calculators.append(
                ConfidenceInterval(x_train=x_train,
                                   y_train=y_train,
                                   conf_level=confidence_level))

    def add_calculator_entropy(self, n_classes):
        """
        Entropy Calculator (CM2 from paper)
        :param n_classes: number of classes of the label
        """
        self.trust_calculators.append(EntropyUncertainty(norm=n_classes))

    def add_calculator_maxprob(self):
        """
        MaxProb Calculator
        """
        self.trust_calculators.append(MaxProbUncertainty())

    def add_calculator_bayes(self, x_train, y_train, n_classes):
        """
        External Trust Calculator using Bayes (CM3 in the paper)
        :param x_train: features in the train set
        :param y_train: labels in the train set
        :param n_classes: number of classes of the label
        """
        self.add_calculator_external(classifier=GaussianNB(), x_train=x_train, y_train=y_train, n_classes=n_classes)

    def add_calculator_external(self, classifier, n_classes, x_train, y_train=None):
        """
        External Trust Calculator (CM3 in the paper)
        :param classifier: classifier to be used as del_clf
        :param x_train: features in the train set
        :param y_train: labels in the train set (if needed)
        :param n_classes: number of classes of the label
        """
        if isinstance(classifier, BaseDetector):
            self.trust_calculators.append(
                ExternalUnsupervisedUncertainty(del_clf=classifier, x_train=x_train, norm=n_classes))
        else:
            self.trust_calculators.append(
                ExternalSupervisedUncertainty(del_clf=classifier, x_train=x_train, y_train=y_train, norm=n_classes))

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

    def add_calculator_agreement(self, clf_set, x_train, y_train=None):
        self.trust_calculators.append(AgreementUncertainty(clf_set=clf_set, x_train=x_train, y_train=y_train))

    def add_calculator_proximity(self, x_train, n_iterations=10, range=0.1, weighted=False):
        self.trust_calculators.append(ProximityUncertainty(x_train, n_iterations, range, weighted))

    def add_calculator_featurebagging(self, x_train, y_train, n_baggers=50, bag_type='sup'):
        self.trust_calculators.append(FeatureBagging(x_train, y_train, n_baggers, bag_type))

    def add_calculator_recloss(self, x_train, tag='simple'):
        """
        External Trust Calculator using Bayes (CM3 in the paper)
        :param x_train: features in the train set
        :param tag: tagstring to initialize autoencoder
        """
        self.trust_calculators.append(ReconstructionLoss(x_train=x_train, enc_tag=tag))

    def predict_set_misclassifications(self, data_set, classifier, y_proba=None, verbose=True, as_pandas=True):
        trust_set = self.compute_set_trust(data_set, classifier, y_proba, verbose, as_pandas)
        return self.predict_misclassifications(trust_set)

    def predict_data_misclassifications(self, data_point, classifier, verbose=True, as_pandas=True):
        trust_data = self.compute_data_trust(data_point, classifier, verbose, as_pandas)
        return self.predict_misclassifications(trust_data)

    def predict_misclassifications(self, trust_set):
        if self.binary_adjudicator is not None:
            sp_df = copy.deepcopy(trust_set)
            if isinstance(trust_set, pandas.DataFrame):
                x_test = sp_df.to_numpy()
            else:
                x_test = sp_df
            predictions = self.binary_adjudicator.predict(x_test)
            sp_df["pred"] = predictions

        else:
            print("Need to load a model for binary adjudication first")

        return sp_df, self.binary_adjudicator

    def load_model(self, model_tag, x_train, y_train=None, label_names=[0, 1]):
        if os.path.exists(self.models_folder):
            if model_tag in self.get_available_models():
                model_folder = self.models_folder + str(model_tag) + "/"
                self.binary_adjudicator = joblib.load(model_folder + "binary_adj_model.joblib")
                print("Loaded Binary Adjudicator '" + get_classifier_name(self.binary_adjudicator) + "'")
                u_calcs = read_calculators(model_folder)
                self.trust_calculators = []
                for uc_tag in u_calcs:
                    params = u_calcs[uc_tag]
                    calculator_name = params["calculator_class"]
                    if "Entropy" in calculator_name:
                        calc = EntropyUncertainty(norm=len(label_names))
                    elif "MaxProb" in calculator_name:
                        calc = MaxProbUncertainty()
                    elif "Neighbors" in calculator_name:
                        calc = NeighborsUncertainty(x_train=x_train, y_train=y_train,
                                                    k=params["n_neighbors"], labels=label_names)
                    elif "ExternalSupervised" in calculator_name:
                        del_clf = joblib.load(model_folder + uc_tag + "_del_clf.joblib")
                        calc = ExternalSupervisedUncertainty(del_clf=del_clf, x_train=x_train, y_train=y_train,
                                                             norm=len(label_names))
                    elif "ExternalUnsupervised" in calculator_name:
                        del_clf = joblib.load(model_folder + uc_tag + "_del_clf.joblib")
                        calc = ExternalUnsupervisedUncertainty(del_clf=del_clf, x_train=x_train, norm=len(label_names))
                    elif ".CombinedUncertainty" in calculator_name:
                        del_clf = joblib.load(model_folder + uc_tag + "_del_clf.joblib")
                        calc = CombinedUncertainty(del_clf=del_clf, x_train=x_train, y_train=y_train,
                                                   norm=len(label_names))
                    elif "MultiCombinedUncertainty" in calculator_name:
                        del_clfs = []
                        clf_files = fnmatch.filter(os.listdir(model_folder), uc_tag + '*.joblib')
                        clf_files.sort(reverse=False)
                        for clf_name in clf_files:
                            del_clf = joblib.load(model_folder + clf_name)
                            del_clfs.append(del_clf)
                        calc = MultiCombinedUncertainty(clf_set=del_clfs, x_train=x_train, y_train=y_train,
                                                        norm=len(label_names))
                    elif "AgreementUncertainty" in calculator_name:
                        del_clfs = []
                        clf_files = fnmatch.filter(os.listdir(model_folder), uc_tag + '*.joblib')
                        clf_files.sort(reverse=False)
                        for clf_name in clf_files:
                            del_clf = joblib.load(model_folder + clf_name)
                            del_clfs.append(del_clf)
                        calc = AgreementUncertainty(clf_set=del_clfs, x_train=x_train)
                    elif "ConfidenceInterval" in calculator_name:
                        calc = ConfidenceInterval(conf_level=params["confidence_level"],
                                                  x_train=x_train, y_train=y_train)
                    elif "ProximityUncertainty" in calculator_name:
                        calc = ProximityUncertainty(x_train=x_train, artificial_points=params["artificial_points"],
                                                    range_wideness=params["range"], weighted=params["weighted"])
                    elif "FeatureBagging" in calculator_name:
                        calc = FeatureBagging(x_train=x_train, y_train=y_train,
                                              n_baggers=params["n_baggers"], bag_type=params["bag_type"])
                    elif "ReconstructionLoss" in calculator_name:
                        calc = ReconstructionLoss(x_train=x_train, enc_tag=params["enc_tag"])
                    else:
                        calc = None
                    if calc is not None:
                        self.trust_calculators.append(calc)

            else:
                print("Model '" + str(model_tag) + "' does not exist")
        else:
            print("Models folder '" + self.models_folder + "' does not exist")

        return self.binary_adjudicator

    def get_available_models(self):
        """
        Returns the models available in the SPROUT repository
        :return: list of strings
        """
        return [f.name for f in os.scandir(self.models_folder) if f.is_dir()]

    def save_object(self, obj_folder):
        # Save general info about calculators into a unique file
        with open(obj_folder + "uncertainty_calculators.txt", 'w') as f:
            f.write('# File that lists the calculators used to build this SPROUT object\n')
            for i in range(0, len(self.trust_calculators)):
                f.write('%s: %s\n' % (str(i+1), self.trust_calculators[i].full_uncertainty_calculator_name()))

        # Saving a file for each uncertainty calculator
        pd = {}
        for i in range(0, len(self.trust_calculators)):
            params_dict = self.trust_calculators[i].save_params(obj_folder + "/", "uncertainty_calculator_" + str(i+1))
            if params_dict is None:
                params_dict = {}
            params_dict["calculator_class"] = get_full_class_name(self.trust_calculators[i].__class__)
            pd["uncertainty_calculator_" + str(i+1)] = params_dict
        with open(obj_folder + "uncertainty_calculator_params.csv", 'w') as f:
            f.write('uncertainty_calculator,param_name,param_value\n')
            for u_calc in pd:
                params_dict = pd[u_calc]
                for key, value in params_dict.items():
                    f.write('%s,%s,%s\n' % (str(u_calc), str(key), str(value)))
