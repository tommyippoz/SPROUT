import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.tab_model import TabNetClassifier
from autogluon.tabular import TabularPredictor


class Classifier:

    def __init__(self, model):
        self.model = model
        self.trained = False


    def fit(self, x_train, y_train):
        self.model.fit(x_train.values, y_train)
        self.trained = True

    def is_trained(self):
        return self.trained

    def predict_class(self, x_test):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        return self.model.predict(x_test)

    def predict_prob(self, x_test):
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

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class UnsupervisedClassifier(Classifier):

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

    def __init__(self):
        Classifier.__init__(self, XGBClassifier(use_label_encoder=False))

    def classifier_name(self):
        return "XGBoost"


class TabNet(Classifier):

    def __init__(self):
        Classifier.__init__(self, TabNetClassifier())

    def fit(self, x_train, y_train):
        self.model.fit(X_train=x_train.to_numpy(), y_train=y_train, eval_metric=['auc'])
        self.trained = True

    def classifier_name(self):
        return "TabNet"


class AutoGluon(Classifier):
    """
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


    def __init__(self, feature_names, label_name, clf_name):
        Classifier.__init__(self, TabularPredictor(label=label_name))
        self.label_name = label_name
        self.feature_names = feature_names
        self.clf_name = clf_name

    def fit(self, x_train, y_train):
        df = pd.DataFrame(data=x_train, columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train_data=df, hyperparameters={self.clf_name:{}})
        self.trained = True

    def predict_class(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict(df)

    def predict_prob(self, x_test):
        df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict_proba(df)

    def classifier_name(self):
        return "AutoGluon"


class FastAI(AutoGluon):

    def __init__(self, feature_names, label_name):
        AutoGluon.__init__(self, feature_names, label_name, "FASTAI")

    def classifier_name(self):
        return "FastAI"


class GBM(AutoGluon):

    def __init__(self, feature_names, label_name):
        AutoGluon.__init__(self, feature_names, label_name, "GBM")

    def classifier_name(self):
        return "GBM"

class MXNet(AutoGluon):

    def __init__(self, feature_names, label_name):
        AutoGluon.__init__(self, feature_names, label_name, "NN")

    def classifier_name(self):
        return "MXNet"


class ADABoostClassifier(Classifier):

    def __init__(self, n_trees):
        Classifier.__init__(self, AdaBoostClassifier(n_estimators=n_trees))

    def classifier_name(self):
        return "XGBoost"


class DecisionTree(Classifier):

    def __init__(self, depth):
        Classifier.__init__(self, DecisionTreeClassifier(max_depth=depth))
        self.depth = depth

    def classifier_name(self):
        return "DecisionTree(depth=" + str(self.depth) + ")"


class KNeighbors(Classifier):

    def __init__(self, k):
        Classifier.__init__(self, KNeighborsClassifier(n_neighbors=k))
        self.k = k

    def classifier_name(self):
        return str(self.k) + "NearestNeighbors"


class LDA(Classifier):

    def __init__(self):
        Classifier.__init__(self, LinearDiscriminantAnalysis())

    def classifier_name(self):
        return "LDA"


class LogisticReg(Classifier):

    def __init__(self):
        Classifier.__init__(self,
                            LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000))

    def classifier_name(self):
        return "LogisticRegression"


class Bayes(Classifier):

    def __init__(self):
        Classifier.__init__(self, GaussianNB())

    def classifier_name(self):
        return "NaiveBayes"


class RandomForest(Classifier):

    def __init__(self, trees):
        Classifier.__init__(self, RandomForestClassifier(n_estimators=trees))
        self.trees = trees

    def classifier_name(self):
        return "RandomForest(trees=" + str(self.trees) + ")"


class SupportVectorMachine(Classifier):

    def __init__(self, kernel, degree):
        Classifier.__init__(self, SVC(kernel=kernel, degree=degree, probability=True, max_iter=10000))
        self.kernel = kernel
        self.degree = degree

    def classifier_name(self):
        return "SupportVectorMachine(kernel=" + str(self.kernel) + ")"


class NeuralNetwork(Classifier):

    def __init__(self, num_input, num_classes):
        self.model = Sequential()
        self.model.add(Dense(num_input, input_shape=(num_input,), activation='relu'))
        self.model.add(Dense(num_input * 10, activation='relu'))
        self.model.add(Dense(num_input * 10, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))

    def fit(self, x_train, y_train):
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, to_categorical(y_train), batch_size=64, epochs=10, verbose=0)
        self.model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def predict_class(self, x_test):
        array_proba = np.asarray(self.model.predict(x_test))
        predictions = np.zeros((len(array_proba),), dtype=int)
        for i in range(len(array_proba)):
            predictions[i] = np.argmax(array_proba[i], axis=0)
        return predictions

    def predict_prob(self, X_test):
        return np.asarray(self.model.predict(X_test))

    def classifier_name(self):
        return "NeuralNetwork"
