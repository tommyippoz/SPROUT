import pandas
import sklearn

from examples.AutoGluonClassifier import AutoGluonClassifier
from quail import quail_utils
from quail.QuailInstance import QuailInstance
from utils.dataset_utils import load_DIGITS


if __name__ == '__main__':
    """
    Main to calculate confidence measures for AutoGluon classifiers using DIGITS dataset
    """

    # Reading sample dataset (DIGITS)
    x_train, x_test, y_train, y_test, label_names, feature_names = load_DIGITS(as_pandas=True)

    print("Preparing Trust Calculators...")

    # Building QUAIL instance and adding Entropy, Bayesian and Neighbour-based Calculators
    quail = QuailInstance()
    quail.add_calculator_entropy(n_classes=len(label_names))
    quail.add_calculator_bayes(x_train=x_train, y_train=y_train, n_classes=len(label_names))
    quail.add_calculator_SHAP(x_train=x_train, feature_names=feature_names)

    # AutoGluon Parameters, clf_name in
    #     ‘GBM’ (LightGBM)
    #     ‘CAT’ (CatBoost)
    #     ‘XGB’ (XGBoost)
    #     ‘RF’ (random forest)
    #     ‘XT’ (extremely randomized trees)
    #     ‘KNN’ (k - nearest neighbors)
    #     ‘LR’ (linear regression)
    #     ‘NN’ (neural network with MXNet backend)
    #     ‘FASTAI’ (neural network with FastAI backend)


    # Building and exercising AutoGluon classifier using Wrapper (for compatibility with SHAP / LIME)
    clf_name = 'GBM'
    classifier = AutoGluonClassifier(feat_names=feature_names, clf_name=clf_name)
    classifier.fit(x_train=x_train, y_train=y_train)
    y_pred = classifier.predict(x_test=x_test)
    y_proba = classifier.predict_proba(x_test=x_test)
    print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Initializing QUAIL dataset for output
    out_df = quail_utils.build_QUAIL_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with QUAIL
    q_df = quail.compute_set_trust(data_set=x_test, y_proba=y_proba, classifier=classifier)
    out_df = pandas.concat([out_df, q_df], axis=1)

    # Printing Dataframe
    out_df.to_csv('my_quail_gluon_df.csv', index=False)
