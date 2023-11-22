import numpy
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier

from sprout.SPROUTObject import SPROUTObject
from sprout.utils import sprout_utils
from sprout.utils.dataset_utils import load_MNIST
from sprout.utils.general_utils import current_ms

MODELS_FOLDER = "../models/"
MODEL_TAG = "test_sup"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for sklearn classifiers using MNIST dataset
    To be updated for the thesis
    """

    # Reading sample dataset (MNIST). Loads data as images
    x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST(flatten=False, row_limit=5000)

    # Reading sample dataset (MNIST). Loads data as images which are going to be linearized (flatten = true)
    # It is required to feed data to SPROUT
    x_train_lin, x_test_lin, y_train_lin, y_test_lin, label_names, feature_names = load_MNIST(flatten=True, row_limit=5000)

    # Here you should initialize and fit your classifier
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(x_train_lin, y_train_lin)

    # Loading SPROUT wrapper for supervised learning
    # The name of the model to be used to detect misclassifications is contained in MODEL_TAG
    # You can use any model tag, but I suggest 'all_sup_fast' or 'all_sup_fast_2'
    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sprout_obj.load_model(model_tag=MODEL_TAG, clf=classifier,
                          x_train=x_train_lin, y_train=y_train_lin, label_names=label_names)

    # This is for predicting class labels for the images in the test set using your classifier
    # Theoretically (?), you should not need to change that
    start_pred = current_ms()
    y_pred = classifier.predict(x_test_lin)
    clf_time = (current_ms() - start_pred) / len(y_test)
    y_proba = classifier.predict_proba(x_test_lin)
    clf_misc = numpy.asarray(y_pred != y_test)
    clf_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Fit and Prediction completed with Accuracy: " + str(clf_acc))

    # Calculating Trust Measures with SPROUT
    # The classifier is needed to compute the confidence measure that uses the nearest neighbours (UM8 in the paper)
    # We may think of building a misclassification detector that does not need that UM8.
    # This would mean that the classifier may not be required here
    # (this is to implement only if this version crashes and I don't find easy and quick bugfixes)
    out_df = sprout_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)
    start_pred = current_ms()
    sp_df, adj = sprout_obj.predict_set_misclassifications(data_set=x_test_lin, y_proba=y_proba, classifier=classifier)
    sprout_time = (current_ms() - start_pred) / len(y_test)
    sprout_pred = sp_df["pred"].to_numpy()

    # Computing SPROUT Metrics for output
    # o_rate is phi
    # sp_acc is alpha_w
    # 1-sp_acc-o_rate is epsilon_w
    o_rate = numpy.average(sprout_pred)
    susp_misc = sum(sprout_pred*clf_misc) / sum(clf_misc)
    sp_acc = numpy.average((1-clf_misc)*(1-sprout_pred))

    print('\n------------  SPROUT Report  ------------------\n')
    print('Regular classifier has accuracy of %.3f and %.3f misclassifications' % (clf_acc, 1-clf_acc))
    print('SPROUT suspects %.3f of misclassifications' % (susp_misc))
    print('Classifier wrapped with SPROUT has %.3f accuracy, %.3f omission rate, '
          'and %.3f residual misclassifications' % (sp_acc, o_rate, 1-sp_acc-o_rate))
    print('Prediction Time of the regular classifier %.3f ms per item, with SPROUT: %.3f per item'
          % (clf_time, sprout_time))

    # Printing Dataframe to file (may be needed for debug or to understand results)
    out_df = pandas.concat([out_df, sp_df.drop(columns=["pred"])], axis=1)
    out_df["clf_pred"] = y_pred
    out_df["true_label"] = y_test
    out_df["sprout_omit (binary confidence score)"] = sp_df["pred"]
    out_df.to_csv('unifi_sprout_output.csv', index=False)
