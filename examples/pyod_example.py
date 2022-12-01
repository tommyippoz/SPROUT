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
from sprout.utils.dataset_utils import process_binary_tabular_dataset

MODELS_FOLDER = "../models/"
MODEL_TAG = "dsn_uns"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for SKLearn classifiers using NSL-KDD dataset from
    https://www.kaggle.com/datasets/hassan06/nslkdd
    """

    # Reading sample dataset (NSL-KDD)
    x_train, x_test, y_train, y_test, label_names, feature_names = \
        process_binary_tabular_dataset(dataset_name="input_folder/NSLKDD.csv", label_name="multilabel")

    print("Preparing Trust Calculators...")

    # Building SPROUT instance and adding Entropy, Bayesian and Neighbour-based Calculators
    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sprout_obj.load_model(model_tag=MODEL_TAG, x_train=x_train, y_train=y_train, label_names=label_names)

    # Building and exercising SKLearn classifier
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)
    print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Initializing SPROUT dataset for output
    out_df = sprout_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with SPROUT
    q_df = sprout_obj.compute_set_trust(data_set=x_test, classifier=classifier)
    out_df = pandas.concat([out_df, q_df], axis=1)
    correlations(out_df)

    # Printing Dataframe
    out_df.to_csv('sklearn_sprout_example.csv', index=False)
