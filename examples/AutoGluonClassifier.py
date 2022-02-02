import numpy
import pandas as pd
from autogluon.tabular import TabularPredictor


class AutoGluonClassifier:
    """
    Wrapper for classifiers taken from AutoGluon library - Needed for the integration with SHAP / LIME
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

    def __init__(self, feat_names, clf_name, metric="accuracy", label_name="label"):
        self.model = TabularPredictor(label=label_name, eval_metric=metric)
        self.label_name = label_name
        self.feature_names = feat_names
        self.clf_name = clf_name
        self.feature_importance = []

    def fit(self, x_train, y_train):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """
        df = pd.DataFrame(data=x_train.copy(), columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train_data=df, hyperparameters={self.clf_name:{}})
        self.feature_importance = self.model.feature_importance(df)

    def feature_importances(self):
        """
        Calculates feature importances of an AutoGluon model
        :return: ndarray of feature importances
        """
        importances = []
        for feature in self.feature_names:
            if feature in self.feature_importance.importance.index.tolist():
                importances.append(abs(self.feature_importance.importance.get(feature)))
            else:
                importances.append(0.0)
        return numpy.asarray(importances)

    def predict(self, x_test):
        """
        Predicts classes for items in a dataset
        :param x_test: the test dataset
        :return: classes for items in x_test
        """
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        """
        Predicts probabilities for items in a dataset
        :param x_test: the test dataset
        :return: probabilities for items in x_test
        """
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict_proba(df, as_pandas=False)

    def classifier_name(self):
        """
        Returns the classifier name
        :return: the classifier name
        """
        return "AutoGluon(" + self.clf_name + ")"
