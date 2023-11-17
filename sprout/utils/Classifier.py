import copy

import numpy
import pandas
import pandas as pd
import pyod
import sklearn
from autogluon.tabular import TabularPredictor
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.so_gaal import SO_GAAL
from pyod.models.suod import SUOD
from pyod.models.vae import VAE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sprout.utils.general_utils import current_ms

# ---------------------------------- SUPPORT METHODS ------------------------------------

def get_classifier_name(clf):
    if isinstance(clf, Classifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def get_feature_importance(clf):
    if isinstance(clf, Classifier):
        return clf.feature_importances()
    else:
        return clf.feature_importances_


def build_classifier(classifier, x_train, y_train, x_test, y_test, verbose=True):
    """
    Builds and Exercises a Classifier
    :param classifier: classifier object
    :param x_train: train features
    :param y_train: train label
    :param x_test: test features
    :param y_test: test label
    :param verbose: True if there should be console output
    :return: probabilities and predictions of the classifier
    """
    if verbose:
        print("\nBuilding classifier: " + get_classifier_name(classifier))

    if isinstance(x_train, pandas.DataFrame):
        train_data = x_train.to_numpy()
    else:
        train_data = x_train

    # Fitting classifier
    start_ms = current_ms()
    if isinstance(classifier, pyod.models.base.BaseDetector):
        classifier.fit(train_data)
    else:
        classifier.fit(train_data, y_train)
    train_ms = current_ms()

    # Test features have to be a numpy array
    if isinstance(x_test, pandas.DataFrame):
        test_data = x_test.to_numpy()
    else:
        test_data = x_test

    # Predicting labels
    y_pred = classifier.predict(test_data)
    test_time = current_ms() - train_ms

    # Predicting probabilities
    y_proba = classifier.predict_proba(test_data)
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.to_numpy()

    if verbose:
        print(get_classifier_name(classifier) + " train/test in " + str(train_ms - start_ms) + "/" +
              str(test_time) + " ms with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    return y_proba, y_pred


def choose_classifier(clf_name, features, y_label, metric, contamination=None):
    if contamination is not None and contamination > 0.5:
        contamination = 0.5
    if clf_name in {"XGB", "XGBoost"}:
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif clf_name in {"DT", "DTree", "DecisionTree"}:
        return DecisionTreeClassifier()
    elif clf_name in {"KNN", "knn", "kNN", "KNeighbours"}:
        return KNeighbors(k=11)
    elif clf_name in {"SVM"}:
        return BaggingClassifier(SVC(gamma='auto', probability=True), max_samples=0.1, n_estimators=10)
    elif clf_name in {"LDA"}:
        return LinearDiscriminantAnalysis()
    elif clf_name in {"Regression", "LogisticRegression", "LR"}:
        return LogisticReg()
    elif clf_name in {"RF", "RandomForest"}:
        return RandomForestClassifier(n_estimators=10)
    elif clf_name in {"TabNet", "Tabnet", "TN"}:
        return TabNet(metric="auc", verbose=2)
    elif clf_name in {"FAI", "FastAI", "FASTAI", "fastai"}:
        return FastAI(label_name=y_label, metric=metric)
    elif clf_name in {"GBC", "GradientBoosting"}:
        return GradientBoostingClassifier(n_estimators=50)
    elif clf_name in {"COPOD"}:
        return UnsupervisedClassifier(COPOD(contamination=contamination))
    elif clf_name in {"ECOD"}:
        return UnsupervisedClassifier(ECOD(contamination=contamination))
    elif clf_name in {"HBOS"}:
        return UnsupervisedClassifier(HBOS(contamination=contamination, n_bins=30))
    elif clf_name in {"MCD"}:
        return UnsupervisedClassifier(MCD(contamination=contamination))
    elif clf_name in {"PCA"}:
        return UnsupervisedClassifier(PCA(contamination=contamination))
    elif clf_name in {"CBLOF"}:
        return UnsupervisedClassifier(CBLOF(contamination=contamination, alpha=0.75, beta=3))
    elif clf_name in {"OCSVM", "1SVM"}:
        return UnsupervisedClassifier(OCSVM(contamination=contamination))
    elif clf_name in {"uKNN"}:
        return UnsupervisedClassifier(KNN(contamination=contamination, n_neighbors=3))
    elif clf_name in {"LOF"}:
        return UnsupervisedClassifier(LOF(contamination=contamination, n_neighbors=5))
    elif clf_name in {"INNE"}:
        return UnsupervisedClassifier(INNE(contamination=contamination))
    elif clf_name in {"FastABOD", "ABOD", "FABOD"}:
        return UnsupervisedClassifier(ABOD(contamination=contamination, method='fast', n_neighbors=7))
    elif clf_name in {"COF"}:
        return UnsupervisedClassifier(COF(contamination=contamination, n_neighbors=9))
    elif clf_name in {"IFOREST", "IForest"}:
        return UnsupervisedClassifier(IForest(contamination=contamination))
    elif clf_name in {"LODA"}:
        return UnsupervisedClassifier(LODA(contamination=contamination, n_bins=100))
    elif clf_name in {"VAE"}:
        return UnsupervisedClassifier(VAE(contamination=contamination))
    elif clf_name in {"SO_GAAL"}:
        return UnsupervisedClassifier(SO_GAAL(contamination=contamination))
    elif clf_name in {"LSCP"}:
        return UnsupervisedClassifier(LSCP(contamination=contamination,
                                           detector_list=[MCD(), COPOD(), HBOS()]))
    elif clf_name in {"SUOD"}:
        return UnsupervisedClassifier(SUOD(contamination=contamination,
                                           base_estimators=[MCD(), COPOD(), HBOS()]))
    else:
        pass


def auto_bag_rate(n_features):
    """
    Method used to automatically devise a rate to include features in bagging
    :param n_features: number of features in the dataset
    :return: the rate of features to bag
    """
    if n_features < 20:
        bag_rate = 0.8
    elif n_features < 50:
        bag_rate = 0.7
    elif n_features < 100:
        bag_rate = 0.6
    else:
        bag_rate = 0.5
    return bag_rate


def predict_uns_proba(uns_clf, test_features):
    proba = uns_clf.predict_proba(test_features)
    pred = numpy.argmax(proba, axis=1)
    for i in range(len(pred)):
        min_p = min(proba[i])
        max_p = max(proba[i])
        proba[i][pred[i]] = max_p
        proba[i][1 - pred[i]] = min_p
    return proba


class Classifier(BaseEstimator):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, classifier):
        """
        Constructor of a generic Classifier
        :param model: model to be used as Classifier
        """
        self.classifier = classifier
        self.trained = False
        self._estimator_type = "classifier"
        self.classes_ = None
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, x_train, y_train=None):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """
        if y_train is not None:
            if isinstance(x_train, pd.DataFrame):
                self.classifier.fit(x_train.to_numpy(), y_train)
            else:
                self.classifier.fit(x_train, y_train)
            self.classes_ = numpy.unique(y_train)
        else:
            if isinstance(x_train, pd.DataFrame):
                self.classifier.fit(x_train.to_numpy())
            else:
                self.classifier.fit(x_train)
            self.classes_ = 2
        self.feature_importances_ = self.get_feature_importances()
        self.trained = True

    def is_trained(self):
        """
        Flags if train was executed
        :return: True if trained, False otherwise
        """
        return self.trained

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        return self.classifier.predict(x_test)

    def predict_proba(self, x_test):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self.classifier.predict_proba(x_test)

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: maximum probability for each data item as default
        """
        proba = self.classifier.predict_proba(x_test)
        return numpy.argmax(proba, axis=1)

    def get_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        # For most SKLearn algorithms
        if hasattr(self.classifier, 'feature_importances_') and self.classifier.feature_importances_ is not None:
            return self.classifier.feature_importances_
        # For statistical algorithms such as regression and LDA/QDA
        elif hasattr(self.classifier, 'coef_') and self.classifier.coef_ is not None:
            return numpy.sum(numpy.absolute(self.classifier.coef_), axis=0)
        else:
            return self.get_feature_importances()

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class UnsupervisedClassifier(Classifier):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, classifier, contamination: float = None):
        Classifier.__init__(self, classifier)
        self.contamination = contamination
        self.name = classifier.__class__.__name__

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: value if algorithm is from framework PYOD
        """
        return self.classifier.predict_confidence(x_test)

    def classifier_name(self):
        return self.name


class TabNet(Classifier):
    """
    Wrapper for the torch.tabnet algorithm
    """

    def __init__(self, metric=None, verbose=0):
        Classifier.__init__(self, TabNetClassifier(verbose=verbose))
        self.metric = metric
        self.verbose = verbose

    def fit(self, x_train, y_train):
        if isinstance(x_train, pandas.DataFrame):
            x_train = x_train.to_numpy()
        if self.metric is None:
            self.classifier.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024, eval_metric=['auc'],
                                patience=2)
        else:
            self.classifier.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024,
                                eval_metric=[self.metric], patience=2)
        self.classes_ = numpy.unique(y_train)
        self.feature_importances_ = self.get_feature_importances()
        self.trained = True

    def predict(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.classifier.predict(x_test)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.classifier.predict_proba(x_test)

    def classifier_name(self):
        return "TabNet"


class AutoGluon(Classifier):
    """
    Wrapper for classifiers taken from Gluon library
    clf_name options are
    ‘GBM’ (LightGBM)
    ‘CAT’ (CatBoost)
    ‘XGB’ (XGBoost)
    ‘RF’ (random forest)
    ‘XT’ (extremely randomized trees)
    ‘KNN’ (k-nearest neighbors)
    ‘LR’ (linear regression)
    ‘NN’ (neural network with MXNet backend)
    ‘FASTAI’ (neural network with FastAI backend)
    """

    def __init__(self, label_name, clf_name, metric, verbose=0):
        Classifier.__init__(self, TabularPredictor(label=label_name, eval_metric=metric, verbosity=verbose))
        self.label_name = label_name
        self.clf_name = clf_name
        self.metric = metric
        self.verbose = verbose
        self.feature_importance = []

    def fit(self, x_train, y_train):
        df = pd.DataFrame(data=x_train.copy(), columns=['col' + str(i) for i in range(0, x_train.shape[1])])
        df[self.label_name] = y_train
        self.classifier.fit(train_data=df, hyperparameters={self.clf_name: {}})
        self.feature_importances_ = self.classifier.feature_importance(df)
        self.classes_ = numpy.unique(y_train)
        self.trained = True

    def get_feature_importances(self):
        return self.feature_importances_

    def predict(self, x_test):
        df = pd.DataFrame(data=x_test, columns=['col' + str(i) for i in range(0, x_test.shape[1])])
        return self.classifier.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        df = pd.DataFrame(data=x_test, columns=['col' + str(i) for i in range(0, x_test.shape[1])])
        return self.classifier.predict_proba(df, as_pandas=False)

    def classifier_name(self):
        return "AutoGluon"


class FastAI(AutoGluon):
    """
    Wrapper for the gluon.FastAI algorithm
    """

    def __init__(self, label_name, metric, verbose=0):
        AutoGluon.__init__(self, label_name, "FASTAI", metric, verbose)

    def classifier_name(self):
        return "FastAI"


class GBM(AutoGluon):
    """
    Wrapper for the gluon.LightGBM algorithm
    """

    def __init__(self, label_name, metric):
        AutoGluon.__init__(self, label_name, "GBM", metric)

    def classifier_name(self):
        return "GBM"


class MXNet(AutoGluon):
    """
    Wrapper for the gluon.MXNet algorithm (to be debugged)
    """

    def __init__(self, label_name):
        AutoGluon.__init__(self, label_name, "NN")

    def classifier_name(self):
        return "MXNet"


class KNeighbors(Classifier):
    """
    Wrapper for the sklearn.kNN algorithm
    """

    def __init__(self, k):
        super().__init__(self, KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm="kd_tree"))
        self.k = k

    def classifier_name(self):
        return str(self.k) + "NearestNeighbors"


class LogisticReg(Classifier):
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self):
        Classifier.__init__(self, LogisticRegression(solver='sag',
                                                     random_state=0,
                                                     multi_class='ovr',
                                                     max_iter=10000,
                                                     n_jobs=10,
                                                     tol=0.1))

    def classifier_name(self):
        return "LogisticRegression"


class SupportVectorMachine(Classifier):
    """
    Wrapper for the sklearn.SVC algorithm
    """

    def __init__(self, kernel, degree):
        Classifier.__init__(self, SVC(kernel=kernel, degree=degree, probability=True, max_iter=10000))
        self.kernel = kernel
        self.degree = degree

    def classifier_name(self):
        return "SupportVectorMachine(kernel=" + str(self.kernel) + ")"


class ConfidenceBoosting(UnsupervisedClassifier):
    """
    Class for creating Unsupervised boosting ensembles
    """

    def __init__(self, estimator, n_base: int = 10, learning_rate: float = None, sampling_ratio: float = None,
                 contamination: float = None, conf_thr: float = None, n_classes: int = 2):
        """
        COnstructor
        :param estimator: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param learning_rate: learning rate for updating dataset weights
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param contamination: percentage of anomalies. TRThis is used to automatically devise conf_thr
        :param conf_thr: threshold of acceptance for confidence scores. Lower confidence means untrustable result
        :param n_classes: number of classes in the problem
        """
        super().__init__(estimator)
        self.classes_ = n_classes
        self.conf_thr = conf_thr
        if contamination is None:
            self.contamination = estimator.contamination
        else:
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

    def fit(self, x_train, y_train=None):
        """
        Training function for the confidence boosting ensemble
        :param y_train: labels of the train set (optional, not required for unsupervised learning)
        :param x_train: train set
        """
        train_n = len(x_train)
        samples_n = int(train_n * self.sampling_ratio)
        weights = numpy.full(train_n, 1 / train_n)
        for i in range(self.n_base):
            # Draw samples
            indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
            sample_x = numpy.asarray([x_train[i] for i in indexes])
            # Train base learner
            learner = copy.deepcopy(self.classifier)
            # if unsupervised, no labels are required for training
            if isinstance(learner, pyod.base.BaseDetector):
                learner.fit(sample_x)
            else:
                learner.fit(sample_x, numpy.asarray([y_train[i] for i in indexes]))
            if self.conf_thr is None:
                self.conf_thr = self.define_proba_thr(x_train, learner)
            self.base_learners.append(learner)
            # Update Weights
            y_proba = predict_uns_proba(learner, x_train)
            y_conf = numpy.max(y_proba, axis=1)
            update_flag = numpy.where(y_conf >= self.conf_thr, 0, 1)
            weights = weights * (1 + self.learning_rate * update_flag)
            weights = weights / sum(weights)
        self.proba_thr = self.define_proba_thr(x_train)
        self.feature_importances_ = self.get_feature_importances()
        self.trained = True

    def define_proba_thr(self, x_train, clf=None, delta=0.01):
        """
        Method for finding a confidence threshold based on the expected contamination (recursive)
        :param x_train: train set
        :param clf: the classifier to use for computing probabilities, or None if self.predict_proba should be used
        :param delta: the tolerance to stop recursion
        :return: a float value to be used as threshold for updating weights in boosting
        """
        target_cont = self.contamination
        if clf is None:
            probs = self.predict_proba(x_train)
        else:
            probs = clf.predict_proba(x_train)
        p_thr = 0.5
        left_bound = 0.5
        right_bound = 1
        actual_cont = numpy.average([1 if p[0] < p_thr else 0 for p in probs])
        while abs(actual_cont - target_cont) > delta and abs(right_bound - left_bound) > 0.01:
            if actual_cont < target_cont:
                left_bound = p_thr
                p_thr = (p_thr + right_bound) / 2
            else:
                right_bound = p_thr
                p_thr = (p_thr + left_bound) / 2
            actual_cont = numpy.average([1 if p[0] < p_thr else 0 for p in probs])
        return p_thr

    def predict_proba(self, x_test):
        """
        Method to compute prediction probabilities (i.e., normalized logits) of a classifier
        :param x_test: the test set
        :return: array of probabilities for each data point and each class
        """
        proba = numpy.zeros((x_test.shape[0], self.classes_))
        for clf in self.base_learners:
            predictions = clf.predict_proba(x_test)
            proba += predictions
        return proba / self.n_base

    def predict_confidence(self, x_test):
        """
        Method to compute the confidence in predictions of a classifier
        :param x_test: the test set
        :return: array of confidence scores
        """
        conf = numpy.zeros(x_test.shape[0])
        for clf in self.base_learners:
            c_conf = clf.predict_confidence(x_test)
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

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :param x_test: the test set
        :return: array of predicted class
        """
        proba = self.predict_proba(x_test)
        return [1 if probs[0] < self.proba_thr else 0 for probs in proba]

    def classifier_name(self):
        return "ConfidenceBooster(" + str(self.n_base) + "-" + str(self.conf_thr) + "-" + str(self.learning_rate) + ")"
