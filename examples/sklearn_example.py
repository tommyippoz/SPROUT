import pandas
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier

from sprout.utils import sprout_utils
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.Classifier import LogisticReg, XGB
from sprout.utils.sprout_utils import correlations
from sprout.utils.dataset_utils import load_MNIST

if __name__ == '__main__':
    """
    Main to calculate confidence measures for SKLearn classifiers using MNIST dataset
    """

    # Reading sample dataset (MNIST)
    x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST()

    print("Preparing Trust Calculators...")

    # Building SPROUT instance and adding Entropy, Bayesian and Neighbour-based Calculators
    sp_obj = SPROUTObject()
    sp_obj.add_all_calculators(x_train=x_train, y_train=y_train, label_names=label_names,
                               combined_clf=XGB(),
                               combined_clfs=[[GaussianNB(), LinearDiscriminantAnalysis(), LogisticReg()],
                                              [GaussianNB(), BernoulliNB(), MultinomialNB(), ComplementNB()],
                                              [DecisionTreeClassifier(), RandomForestClassifier(), XGB()]])

    # Building and exercising SKLearn classifier
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)
    print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Initializing SPROUT dataset for output
    out_df = sprout_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with SPROUT
    q_df = sp_obj.compute_set_trust(data_set=x_test, classifier=classifier)
    out_df = pandas.concat([out_df, q_df], axis=1)
    correlations(out_df)

    # Printing Dataframe
    out_df.to_csv('sklearn_sprout_example.csv', index=False)
