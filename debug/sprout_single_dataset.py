import os

import numpy
import numpy as np
import pandas as pd
import sklearn

from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import choose_classifier, get_classifier_name, build_classifier
from sprout.utils import dataset_utils, sprout_utils
from sprout.utils.general_utils import load_config, clean_name

MODELS_FOLDER = "../models/"
MODEL_TAGS = ['sup_all']
OUTPUT_FOLDER = "./output_folder/"
OUTPUT_LOG_FILE = "grid_sprout.csv"

if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Reading preferences
    dataset_files, d_folder, s_folder, s_clf, u_clf, y_label, limit_rows = load_config("config.cfg")

    with open(OUTPUT_FOLDER + OUTPUT_LOG_FILE, 'w') as f:
        f.write('dataset,classifier,detector_tag,detector_classifier,classifier_acc,detector_acc,detector_mcc,' +
                'SPROUT_availability,SPROUT_accuracy\n')

    if MODEL_TAGS is None:
        MODEL_TAGS = os.listdir(MODELS_FOLDER)

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not dataset_utils.is_image_dataset(dataset_file):
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")
        else:
            print("Processing Dataset " + dataset_file + (
                " - limit " + str(limit_rows) if np.isfinite(limit_rows) else ""))
            if dataset_file.endswith('.csv'):
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_tabular_dataset(dataset_file, y_label, limit_rows, shuffle=True)
            else:
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_image_dataset(dataset_file, limit_rows)

            for classifier_string in s_clf:
                # Building and exercising classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                y_proba, y_pred = build_classifier(classifier, x_train, y_train, x_test, y_test)

                for tag in MODEL_TAGS:
                    print("Loading SPROUT Model for tag '" + str(tag) + "' ...")
                    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
                    sprout_obj.load_model(model_tag=tag, clf=classifier,
                                          x_train=x_train, y_train=y_train, label_names=label_tags)

                    clf_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
                    y_misc = numpy.multiply(y_pred != y_test, 1)

                    # Initializing SPROUT dataset for output
                    out_df = sprout_utils.build_SPROUT_dataset(x_test, y_proba, y_pred, y_test, label_tags)

                    # Calculating Trust Measures with SPROUT
                    sp_df = sprout_obj.compute_set_trust(data_set=x_test, classifier=classifier)
                    sp_df = sp_df.select_dtypes(exclude=['object'])
                    sp_df.reset_index(drop=True, inplace=True)
                    out_df = pd.concat([out_df, sp_df], axis=1)

                    # Predict misclassifications with SPROUT
                    predictions_df, clf = sprout_obj.predict_misclassifications(sp_df)
                    y_pred = predictions_df["pred"].to_numpy()

                    # Printing Dataframe
                    file_out = OUTPUT_FOLDER + clean_name(dataset_file, d_folder) + "_" + \
                               get_classifier_name(classifier) + '.csv'
                    if not os.path.exists(os.path.dirname(file_out)):
                        os.mkdir(os.path.dirname(file_out))
                    out_df["SPROUT_pred"] = y_pred
                    out_df.to_csv(file_out, index=False)
                    print("File '" + file_out + "' Printed")

                    y_true = y_misc
                    [tn, fp], [fn, tp] = sklearn.metrics.confusion_matrix(y_true, y_pred)
                    best_metrics = {"MCC": sklearn.metrics.matthews_corrcoef(y_true, y_pred),
                                    "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
                                    "AUC ROC": sklearn.metrics.roc_auc_score(y_true, y_pred),
                                    "Precision": sklearn.metrics.precision_score(y_true, y_pred),
                                    "Recall": sklearn.metrics.recall_score(y_true, y_pred),
                                    "TP": tp,
                                    "TN": tn,
                                    "FP": fp,
                                    "FN": fn}

                    print("\nMisclassification Detector [" + tag + "]: " + get_classifier_name(clf) +
                          " for classifier '" + get_classifier_name(classifier) + "' "
                                                                                  "has ACC = " + str(
                        best_metrics["Accuracy"]) + ", MCC = " + str(best_metrics["MCC"]))

                    availability = 100.0 * numpy.count_nonzero(y_pred == 0) / len(y_pred)
                    avail_y = y_misc[y_pred == 0]
                    sys_ACC = numpy.count_nonzero(avail_y == 0) / len(avail_y)

                    print("SPROUT Scores with clf = " + get_classifier_name(clf) + " and wrapper = " +
                          get_classifier_name(classifier) + ": av=" + str(availability) + "%, ACC=" + str(sys_ACC) +
                          ", Regular ACC =" + str(clf_acc))

                    with open(OUTPUT_FOLDER + OUTPUT_LOG_FILE, 'a') as f:
                        f.write(dataset_file + "," + get_classifier_name(classifier) + "," + tag + "," +
                                get_classifier_name(clf) + "," + str(clf_acc) + "," +
                                str(best_metrics["Accuracy"]) + "," + str(best_metrics["MCC"]) +
                                "," + str(availability) + "," + str(sys_ACC) + '\n')
