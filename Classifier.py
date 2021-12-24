from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential


class Classifier:

    def __init__(self, X_train, y_train, X_test, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model

    def predict_(self):
        return self.model.fit(self.X_train, self.y_train).predict(self.X_test)

    def predict_proba_(self):
        return self.model.fit(self.X_train, self.y_train).predict_proba(self.X_test)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class GBClassifier(XGBClassifier, Classifier):

    def __init__(self, X_train, y_train, X_test):
        Classifier.__init__(self, X_train, y_train, X_test, XGBClassifier())

    def classifier_name(self):
        return "XGBoost"


class DecisionTree(DecisionTreeClassifier, Classifier):

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
        Classifier.__init__(self, X_train, y_train, X_test, SVC())

    def classifier_name(self):
        return "CSupportVector"


class NeuralNetwork(Classifier):

    def __init__(self, X_train, y_train, X_test):
        self.model = Sequential()
        super().__init__(X_train, y_train, X_test)

    def classifier_name(self):
        return "NeuralNetwork"

