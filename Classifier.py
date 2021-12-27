import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class Classifier:

    def __init__(self, X_train, y_train, X_test, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model

    def predict_class(self):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        return self.model.fit(self.X_train, self.y_train).predict(self.X_test)

    def predict_prob(self):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self.model.fit(self.X_train, self.y_train).predict_proba(self.X_test)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class GBClassifier(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, XGBClassifier())

    def classifier_name(self):
        return "XGBoost"


class DecisionTree(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, DecisionTreeClassifier())

    def classifier_name(self):
        return "DecisionTree"


class KNeighbors(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, KNeighborsClassifier())

    def classifier_name(self):
        return "KNeighbors"


class LDA(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, LinearDiscriminantAnalysis())

    def classifier_name(self):
        return "LDA"


class LogisticReg(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test,
                            LogisticRegression(random_state=0, multi_class='auto', max_iter=10000))

    def classifier_name(self):
        return "LogisticRegression"


class Bayes(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, GaussianNB())

    def classifier_name(self):
        return "Bayes"


class RandomForest(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, RandomForestClassifier())

    def classifier_name(self):
        return "RandomForest"


class CSupportVector(Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, SVC(probability=True))

    def classifier_name(self):
        return "CSupportVector"


class NeuralNetwork(Classifier):

    def __init__(self, X_train, y_train, X_test):
        self.le = LabelEncoder()
        self.y_train = self.le.fit_transform(y_train)
        categorical = to_categorical(self.y_train)
        num_classes = len(categorical[0])
        num_input = len(X_test.values[0])
        self.model = Sequential()
        self.model.add(Dense(num_input, input_shape=(num_input,), activation='relu'))
        self.model.add(Dense(num_input * 10, activation='relu'))
        self.model.add(Dense(num_input * 10, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, categorical, batch_size=64, epochs=10, verbose=0)
        self.model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        self.array_proba = np.asarray(self.model.predict(X_test))
        # super().__init__(X_train, self.y_train, X_test, self.model)

    def predict_class(self):
        predictions = np.zeros((len(self.array_proba),), dtype=int)
        for i in range(len(self.array_proba)):
            predictions[i] = np.argmax(self.array_proba[i], axis=0)
        return self.le.inverse_transform(predictions)

    def predict_prob(self):
        return self.array_proba

    def classifier_name(self):
        return "NeuralNetwork"
