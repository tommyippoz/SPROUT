import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from pytorch_tabnet.tab_model import TabNetClassifier
from autogluon.tabular import TabularPredictor


class Classifier:
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, model):
        """
        Constructor of a generic Classifier
        :param model: model to be used as Classifier
        """
        self.model = model
        self.trained = False

    def fit(self, x_train, y_train):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """
        if isinstance(x_train, pd.DataFrame):
            self.model.fit(x_train.values, y_train)
        else:
            self.model.fit(x_train, y_train)
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
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self.model.predict_proba(x_test)

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return -1

    def feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        return self.model.feature_importances_

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class UnsupervisedClassifier(Classifier):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, classifier, name):
        Classifier.__init__(self, classifier)
        self.name = name

    def fit(self, x_train, y_train):
        self.model.fit(x_train.values)
        self.trained = True

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return self.model.predict_confidence(x_test)

    def classifier_name(self):
        return self.name


class XGB(Classifier):
    """
    Wrapper for the XGBoost  algorithm from xgboost library
    """

    def __init__(self, n_trees=None, metric=None):
        self.metric = metric
        if n_trees is None:
            Classifier.__init__(self, XGBClassifier(use_label_encoder=False))
        else:
            Classifier.__init__(self, XGBClassifier(n_estimators=n_trees, use_label_encoder=False))

    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            self.model.fit(x_train.values, y_train, eval_metric=(self.metric if self.metric is not None else "logloss"))
        else:
            self.model.fit(x_train, y_train, eval_metric=(self.metric if self.metric is not None else "logloss"))
        self.trained = True

    def classifier_name(self):
        return "XGBoost"


class TabNet(Classifier):
    """
    Wrapper for the torch.tabnet algorithm
    """

    def __init__(self, metric=None):
        Classifier.__init__(self, TabNetClassifier())
        self.metric = metric

    def fit(self, x_train, y_train):
        if self.metric is None:
            self.model.fit(X_train=x_train.to_numpy(), y_train=y_train, eval_metric=['auc'])
        else:
            self.model.fit(X_train=x_train.to_numpy(), y_train=y_train, eval_metric=[self.metric])
        self.trained = True

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

    def __init__(self, feature_names, label_name, clf_name, metric):
        Classifier.__init__(self, TabularPredictor(label=label_name, eval_metric=metric))
        self.label_name = label_name
        self.feature_names = feature_names
        self.clf_name = clf_name
        self.feature_importance = []

    def fit(self, x_train, y_train):
        df = pd.DataFrame(data=x_train.copy(), columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train_data=df, hyperparameters={self.clf_name:{}})
        self.feature_importance = self.model.feature_importance(df)
        self.trained = True

    def feature_importances(self):
        return self.feature_importance

    def predict(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict_proba(df, as_pandas=False)

    def classifier_name(self):
        return "AutoGluon"


class FastAI(AutoGluon):
    """
    Wrapper for the gluon.FastAI algorithm
    """

    def __init__(self, feature_names, label_name, metric):
        AutoGluon.__init__(self, feature_names, label_name, "FASTAI", metric)

    def classifier_name(self):
        return "FastAI"


class GBM(AutoGluon):
    """
    Wrapper for the gluon.LightGBM algorithm
    """

    def __init__(self, feature_names, label_name, metric):
        AutoGluon.__init__(self, feature_names, label_name, "GBM", metric)

    def classifier_name(self):
        return "GBM"


class MXNet(AutoGluon):
    """
    Wrapper for the gluon.MXNet algorithm (to be debugged)
    """

    def __init__(self, feature_names, label_name):
        AutoGluon.__init__(self, feature_names, label_name, "NN")

    def classifier_name(self):
        return "MXNet"


class KNeighbors(Classifier):
    """
    Wrapper for the sklearn.kNN algorithm
    """

    def __init__(self, k):
        Classifier.__init__(self, KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm="kd_tree"))
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


class Bayes(Classifier):
    """
    Wrapper for the sklearn.GaussianNB algorithm
    """

    def __init__(self):
        Classifier.__init__(self, GaussianNB())

    def classifier_name(self):
        return "NaiveBayes"


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
