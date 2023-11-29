import copy

import numpy
import pandas
import pandas as pd
import pyod
import sklearn
from autogluon.tabular import TabularPredictor
from pyod.models.abod import ABOD
from pyod.models.base import BaseDetector
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
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
    elif clf_name in {"GNB", "GaussianNB"}:
        return Pipeline([("norm", MinMaxScaler()), ("clf", GaussianNB())])
    elif clf_name in {"MNB", "MultinomialNB"}:
        return Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())])
    elif clf_name in {"Regression", "LogisticRegression", "LR"}:
        return LogisticReg()
    elif clf_name in {"RF", "RandomForest"}:
        return RandomForestClassifier(n_estimators=10)
    elif clf_name in {"TabNet", "Tabnet", "TN"}:
        return TabNet(metric="auc", verbose=0)
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
        return UnsupervisedClassifier(MCD(contamination=contamination, support_fraction=0.9))
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
        return UnsupervisedClassifier(IForest(contamination=contamination, n_estimators=10))
    elif clf_name in {"LODA"}:
        return UnsupervisedClassifier(LODA(contamination=contamination, n_bins=100))
    elif clf_name in {"VAE"}:
        return UnsupervisedClassifier(VAE(contamination=contamination))
    elif clf_name in {"SO_GAAL"}:
        return UnsupervisedClassifier(SO_GAAL(contamination=contamination))
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


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, clf):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        """
        self.clf = clf
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        if y is not None:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0, 1]
        #self.X_ = X
        #self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return self.classes_[numpy.argmax(probas, axis=1)]

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        return self.clf.predict_proba(X)

    def predict_confidence(self, X):
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        elif hasattr(self.clf, 'coef_'):
            return numpy.sum(numpy.absolute(self.clf.coef_), axis=0)
        return []

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__

    def get_params(self, deep=True):
        return {'clf': self.clf}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.clf, parameter, value)
        return self


class UnsupervisedClassifier(Classifier, BaseDetector):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, clf):
        """
        Constructor of a generic UnsupervisedClassifier. Assumes that clf is an algorithm from pyod
        :param clf: pyod algorithm to be used as Classifier
        """
        self.clf = clf
        self.contamination = clf.contamination
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):

        # Store the classes seen during fit + other data
        self.classes_ = [0, 1]
        self.X_ = X
        self.y_ = None

        # Train clf
        self.clf.fit(X)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def decision_function(self, X):
        """
        pyod function to override. Calls the wrapped classifier.
        :param X: test set
        :return: decision function
        """
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes.
        It has to e overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        pred_score = self.decision_function(X)
        probs = numpy.zeros((X.shape[0], 2))
        if isinstance(self.contamination, (float, int)):
            pred_thr = pred_score - self.clf.threshold_
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
        return probs

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return numpy.argmax(probas, axis=1)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__


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
            self.clf.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024, eval_metric=['auc'],
                         patience=2)
        else:
            self.clf.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024,
                         eval_metric=[self.metric], patience=2)
        self.classes_ = numpy.unique(y_train)
        self.feature_importances_ = self.get_feature_importances()

    def get_feature_importances(self):
        return self.clf.feature_importances_

    def predict(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.clf.predict(x_test)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.clf.predict_proba(x_test)

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
        self.feature_names = None

    def fit(self, x_train, y_train):
        path = './AutogluonModels/' + str(current_ms())
        self.classes_ = numpy.unique(y_train)
        self.clf = TabularPredictor(label=self.label_name, eval_metric=self.metric,
                                    path=path, verbosity=self.verbose)
        if self.feature_names is None:
            self.feature_names = ['col' + str(i) for i in range(0, x_train.shape[1])]
        df = pd.DataFrame(data=x_train.copy(), columns=self.feature_names)
        df[self.label_name] = y_train
        self.clf.fit(train_data=df, hyperparameters={self.clf_name: {}})
        self.feature_importances_ = self.clf.feature_importance(df)

        self.trained = True

    def get_feature_importances(self):
        return self.feature_importances_

    def predict(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.clf.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.clf.predict_proba(df, as_pandas=False)

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


class XGB(Classifier):
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self, n_estimators=100):
        Classifier.__init__(self, XGBClassifier(n_estimators=n_estimators))
        self.l_encoder = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        self.classes_ = unique_labels(y)
        self.l_encoder = LabelEncoder()
        y = self.l_encoder.fit_transform(y)

        #self.X_ = X
        #self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def classifier_name(self):
        return "XGBClassifier"


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
