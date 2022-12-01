import pandas
import sklearn

from examples.AutoGluonClassifier import AutoGluonClassifier
from sprout.utils import sprout_utils
from sprout.SPROUTObject import SPROUTObject
from sprout.utils.sprout_utils import correlations
from sprout.utils.dataset_utils import load_DIGITS

MODELS_FOLDER = "../models/"
MODEL_TAG = "dsn_sup_2"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for AutoGluon classifiers using DIGITS dataset
    """

    # Reading sample dataset (DIGITS)
    x_train, x_test, y_train, y_test, label_names, feature_names = load_DIGITS(as_pandas=True)

    # Loading SPROUT wrapper for supervised learning
    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sprout_obj.load_model(model_tag=MODEL_TAG, x_train=x_train, y_train=y_train, label_names=label_names)

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

    # Initializing SPROUT dataset for output
    out_df = sprout_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with SPROUT
    q_df = sprout_obj.compute_set_trust(data_set=x_test, y_proba=y_proba, classifier=classifier)
    out_df = pandas.concat([out_df, q_df], axis=1)
    correlations(out_df)

    # Printing Dataframe
    out_df.to_csv('my_quail_gluon_df.csv', index=False)
