import numpy
import pandas
import sklearn
from pyod.models.copod import COPOD

from sprout.SPROUTObject import SPROUTObject
from sprout.utils import sprout_utils
from sprout.utils.dataset_utils import process_binary_tabular_dataset
from sprout.utils.sprout_utils import correlations

MODELS_FOLDER = "../models/"
MODEL_TAG = "dsn_uns_2"


def compute_omission_metrics(y_test):
    pass


if __name__ == '__main__':
    """
    Main to calculate confidence measures for PYOD classifiers using NSL-KDD dataset from
    https://www.kaggle.com/datasets/hassan06/nslkdd
    """

    # Reading sample dataset (NSL-KDD)
    x_train, x_test, y_train, y_test, label_names, feature_names = \
        process_binary_tabular_dataset(dataset_name="input_folder/UNSW.csv", label_name="multilabel", limit=10000)

    print("Preparing Trust Calculators...")

    # Building SPROUT instance and adding Entropy, Bayesian and Neighbour-based Calculators
    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sprout_obj.load_model(model_tag=MODEL_TAG, x_train=x_train, y_train=y_train, label_names=label_names)

    # Building and exercising SKLearn classifier
    classifier = COPOD()
    classifier.fit(x_train)
    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)
    clf_misc = numpy.asarray(y_pred != y_test)
    clf_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Fit and Prediction completed with Accuracy: " + str(clf_acc))

    # Initializing SPROUT dataset for output
    out_df = sprout_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)

    # Calculating Trust Measures with SPROUT
    sp_df, adj = sprout_obj.predict_set_misclassifications(data_set=x_test, y_proba=y_proba, classifier=classifier)
    sprout_pred = sp_df["pred"].to_numpy()
    p_metrics = compute_omission_metrics(y_test)

    # Computing SPROUT Metrics
    o_rate = numpy.average(sprout_pred)
    susp_misc = sum(sprout_pred * clf_misc) / sum(clf_misc)
    sp_acc = numpy.average((1 - clf_misc) * (1 - sprout_pred))

    print('\n------------  SPROUT Report  ------------------\n')
    print('Regular classifier has accuracy of %.3f and %.3f misclassifications' % (clf_acc, 1 - clf_acc))
    print('SPROUT suspects %.2f of misclassifications' % (100.0*susp_misc))
    print('Classifier wrapped with SPROUT has %.3f accuracy, %.3f omission rate, '
          'and %.3f residual misclassifications' % (sp_acc, o_rate, 1 - sp_acc - o_rate))

    out_df = pandas.concat([out_df, sp_df.drop(columns=["pred"])], axis=1)
    correlations(out_df)

    # Printing Dataframe
    out_df.to_csv('pyod_example.csv', index=False)

