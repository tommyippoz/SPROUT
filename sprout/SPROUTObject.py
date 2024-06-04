import copy
import fnmatch
import os.path

import joblib
import numpy
import pandas
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.naive_bayes import GaussianNB

from sprout.UncertaintyCalculator import EntropyUncertainty, ConfidenceInterval, ExternalSupervisedUncertainty, \
    CombinedUncertainty, MultiCombinedUncertainty, NeighborsUncertainty, ProximityUncertainty, \
    ReconstructionLoss, \
    ExternalUnsupervisedUncertainty, MaxProbUncertainty, AgreementUncertainty, \
    ConfidenceBaggingUncertainty, ConfidenceBoostingUncertainty
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils import general_utils
from sprout.utils.general_utils import get_full_class_name, current_ms
from sprout.utils.sprout_utils import read_calculators, compute_omission_metrics


def exercise_wrapper(model_tag, models_folder, classifier, x_train, y_train, x_test, y_test, label_names, verbose=True):
    # Creating classifier clf
    if verbose:
        print("\nBuilding classifier: " + get_classifier_name(classifier))
    start_ms = current_ms()
    classifier.fit(x_train, y_train)
    train_ms = current_ms()
    y_pred = classifier.predict(x_test)
    test_time = current_ms() - train_ms
    train_time = train_ms - start_ms

    # Loading SPROUT object with a specific tag amongst those existing
    sprout_obj = SPROUTObject(models_folder=models_folder)
    sprout_obj.load_model(model_tag=model_tag, clf=classifier,
                          x_train=x_train, y_train=y_train, label_names=label_names)
    start_ms = current_ms()
    sprout_df, sprout_pred = sprout_obj.exercise(x=x_test, y=y_test, classifier=classifier, verbose=verbose)
    sprout_time = current_ms() - start_ms

    # Computing metrics and printing results
    metrics = compute_omission_metrics(y_test, sprout_pred, y_pred)
    metrics['train_time'] = train_time
    metrics['test_time'] = test_time
    metrics['sprout_time'] = sprout_time
    metrics['clf_name'] = get_classifier_name(classifier)
    metrics['sprout_tag'] = model_tag

    return metrics

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
        if isinstance(data_set, pandas.DataFrame):
            data_set = data_set.to_numpy()
        data_set = numpy.nan_to_num(data_set, nan=0, posinf=0, neginf=0)
        for calculator in self.trust_calculators:
            if verbose:
                print("Calculating Trust Strategy: " + calculator.uncertainty_calculator_name())
            start_ms = general_utils.current_ms()
            trust_scores = calculator.uncertainty_scores(data_set, y_proba, classifier)
            trust_scores = numpy.nan_to_num(trust_scores, nan=-10, posinf=-10, neginf=-10)
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

    def add_calculator_bagging(self, base_clf, x_train, y_train, n_base: int = 10, max_features: float = 0.7,
                               sampling_ratio: float = 0.7, perc_decisors: float = None, n_decisors: int = None,
                               n_classes=2):
        self.trust_calculators.append(
            ConfidenceBaggingUncertainty(base_clf, x_train, y_train, n_base, max_features, sampling_ratio,
                                         perc_decisors, n_decisors, n_classes))

    def add_calculator_boosting(self, base_clf, x_train, y_train, n_base: int = 10, learning_rate: float = None,
                                sampling_ratio: float = 0.5, contamination: float = None, conf_thr: float = 0.8,
                                n_classes=2):
        self.trust_calculators.append(
            ConfidenceBoostingUncertainty(base_clf, x_train, y_train, n_base, learning_rate, sampling_ratio,
                                          contamination, conf_thr, n_classes))

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
            if isinstance(sp_df, pandas.DataFrame):
                x_test = sp_df.select_dtypes(exclude=['object'])
                x_test = x_test.to_numpy()
            else:
                x_test = sp_df
            x_test = numpy.nan_to_num(x_test)
            predictions = self.binary_adjudicator.predict(x_test)
            sp_df["pred"] = predictions

        else:
            print("Need to load a model for binary adjudication first")

        return sp_df, self.binary_adjudicator

    def load_model(self, model_tag, clf, x_train, y_train=None, label_names=[0, 1], load_calculators=True):
        if os.path.exists(self.models_folder):
            if model_tag in self.get_available_models():
                model_folder = self.models_folder + str(model_tag) + "/"
                self.binary_adjudicator = joblib.load(model_folder + "binary_adj_model.joblib")
                print("Loaded Binary Adjudicator '%s' for wrapper '%s'" %
                      (get_classifier_name(self.binary_adjudicator), model_tag))
                if load_calculators:
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
                            calc = ExternalUnsupervisedUncertainty(del_clf=del_clf, x_train=x_train,
                                                                   norm=len(label_names))
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
                        elif "ConfidenceBagging" in calculator_name:
                            calc = ConfidenceBaggingUncertainty(clf=clf, x_train=x_train, y_train=y_train,
                                                                n_base=int(params["n_base"]) if params["n_base"] != 'None' else None,
                                                                max_features=float(params["max_features"]) if params["max_features"] != 'None' else None,
                                                                sampling_ratio=float(params["sampling_ratio"]) if params["sampling_ratio"] != 'None' else None,
                                                                n_decisors=int(params["n_decisors"]) if params["n_decisors"] != 'None' else None,
                                                                n_classes=len(label_names))
                        elif "ConfidenceBoosting" in calculator_name:
                            calc = ConfidenceBoostingUncertainty(clf=clf, x_train=x_train, y_train=y_train,
                                                                n_base=int(params["n_base"]) if params["n_base"] != 'None' else None,
                                                                learning_rate=float(params["learning_rate"]) if params["learning_rate"] != 'None' else None,
                                                                sampling_ratio=float(params["sampling_ratio"]) if params["sampling_ratio"] != 'None' else None,
                                                                contamination=float(params["contamination"]) if params["contamination"] != 'None' else None,
                                                                conf_thr=float(params["conf_thr"]) if params["conf_thr"] != 'None' else None,
                                                                n_classes=len(label_names))
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
                f.write('%s: %s\n' % (str(i + 1), self.trust_calculators[i].full_uncertainty_calculator_name()))

        # Saving a file for each uncertainty calculator
        pd = {}
        for i in range(0, len(self.trust_calculators)):
            params_dict = self.trust_calculators[i].save_params(obj_folder + "/",
                                                                "uncertainty_calculator_" + str(i + 1))
            if params_dict is None:
                params_dict = {}
            params_dict["calculator_class"] = get_full_class_name(self.trust_calculators[i].__class__)
            pd["uncertainty_calculator_" + str(i + 1)] = params_dict
        with open(obj_folder + "uncertainty_calculator_params.csv", 'w') as f:
            f.write('uncertainty_calculator,param_name,param_value\n')
            for u_calc in pd:
                params_dict = pd[u_calc]
                for key, value in params_dict.items():
                    f.write('%s,%s,%s\n' % (str(u_calc), str(key), str(value)))

    def exercise(self, x, y, classifier, verbose=True):
        """
        Exercised the SPROUT wrapper on a test set (x,y). Assumes that the binary adjudicator is ready,
        meaning that the object has already been loaded successfully
        :param x: the test features x_test
        :param y: the test label y_test
        :param classifier: the classifier object
        :param verbose: True if debug information has to be shown
        :return: the pandas.DataFrame containing all data plus an array of predictions of the wrapper
        """
        if isinstance(x, pandas.DataFrame):
            out_df = x.copy()
        else:
            out_df = pandas.DataFrame(data=x)
        out_df.reset_index(drop=True, inplace=True)
        out_df['true_label'] = y
        clf_pred = classifier.predict(x)
        out_df['predicted_label'] = clf_pred
        out_df['is_misclassification'] = numpy.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
        y_proba = classifier.predict_proba(x)
        out_df['probabilities'] = [numpy.array2string(y_proba[i], separator=";") for i in range(len(y_proba))]

        if self.binary_adjudicator is not None:
            # Calculating Trust Measures with SPROUT
            sp_df = self.compute_set_trust(data_set=x, classifier=classifier, verbose=verbose)
            sp_df = sp_df.select_dtypes(exclude=['object'])
            sp_df.reset_index(drop=True, inplace=True)
            out_df = pd.concat([out_df, sp_df], axis=1)

            # Predict misclassifications with SPROUT
            predictions_df, clf = self.predict_misclassifications(sp_df)
            misc_pred = predictions_df["pred"].to_numpy()
            out_df["misc_pred"] = misc_pred
            sprout_pred = numpy.where(misc_pred == 0, clf_pred, None)
            out_df["sprout_pred"] = sprout_pred

            return out_df, sprout_pred
        else:
            print('Unable to load SPROUT model')
            return out_df, clf_pred
