import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier

from sprout.utils import sprout_utils
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.sprout_utils import correlations
from utils.dataset_utils import load_MNIST

if __name__ == '__main__':
    """
    Main to calculate confidence measures for SKLearn classifiers using MNIST dataset
    """

    # Reading sample dataset (MNIST)
    x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST()

    print("Preparing Trust Calculators...")

    # Building SPROUT instance and adding Entropy, Bayesian and Neighbour-based Calculators
    quail = SPROUTObject()
    quail.add_calculator_entropy(n_classes=len(label_names))
    quail.add_calculator_neighbour(x_train=x_train, y_train=y_train, label_names=label_names)

    # Building and exercising SKLearn classifier
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)
    print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Initializing SPROUT dataset for output
    out_df = quail_utils.build_QUAIL_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with SPROUT
    q_df = quail.compute_set_trust(data_set=x_test, classifier=classifier)
    out_df = pandas.concat([out_df, q_df], axis=1)
    correlations(out_df)

    # Printing Dataframe
    out_df.to_csv('my_quail_df.csv', index=False)
