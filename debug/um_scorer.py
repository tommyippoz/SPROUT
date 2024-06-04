import os

import numpy
import pyod
import sklearn
from sklearn.tree import DecisionTreeClassifier

from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import get_classifier_name, choose_classifier
from sprout.utils import dataset_utils
from sprout.utils.general_utils import load_config, clean_name, current_ms
from sprout.utils.sprout_utils import compute_omission_metrics

MODELS_FOLDER = "../models/"
MODEL_TAG = 'sup_all'
DATA_FOLDER = 'input_folder/grids'
OUT_FILE = 'output_folder/grids/grid_stats.csv'
OUTPUT_FOLDER = 'output_folder'

UMs = {'UM1': 'Confidence Interval (0.9/sup)',
       'UM2': 'MaxProb Calculator',
       'UM3': 'Entropy Calculator',
       'UM4': 'External Supervised Calculator (Pipeline/entropy)',
       'UM5': 'Combined Calculator (XGBClassifier)',
       'UM6': 'Multiple Combined Calculator (3 - PeLsLn classifiers)',
       'UM6_stat': 'Multiple Combined Calculator (4 - PePePePe classifiers)',
       'UM6_tree': 'Multiple Combined Calculator (3 - DrRrGr classifiers)',
       'UM7': 'uncertainty calculator on 19 Neighbors_uncertainty',
       'UM8_1': 'Proximity Uncertainty (10/0.1/W)',
       'UM8_2': 'Proximity Uncertainty (20/0.05/W)',
       'UM9': 'AutoEncoder Loss (simple)',
       'UM10': 'ConfidenceBagger(10-6-0.7-0.7)',
       'UM11': 'ConfidenceBooster(10-0.8-2.0-0.5)'}

COMBINATIONS = ['UM1', 'UM2', 'UM3', 'UM4', 'UM5', 'UM6', 'UM6_stat', 'UM6_tree', 'UM7', 'UM8_1',
                'UM8_2', 'UM9', 'UM10', 'UM11', ['UM1', 'UM9'], ['UM2', 'UM3'],
                ['UM4', 'UM5', 'UM6', 'UM6_stat', 'UM6_tree'], ['UM7', 'UM8_1', 'UM8_2'], ['UM10', 'UM11'],
                ['UM1', 'UM9', 'UM2', 'UM3'], ['UM1', 'UM2', 'UM3', 'UM4', 'UM5', 'UM6', 'UM6_stat', 'UM6_tree',
                'UM7', 'UM8_1', 'UM8_2', 'UM9', 'UM10', 'UM11']]

if __name__ == '__main__':
    """
    Main to evaluate the importance of each UM for suspecting misclassifications
    """

    # Generating Input data for training Misclassification Predictors
    if not os.path.exists(DATA_FOLDER):
        print('Unable to find data folder')
        exit(1)

    with open(OUT_FILE, 'w') as f:
        f.write("dataset,clf,ums,alpha,eps,phi,alpha_w,eps_w,phi_c,phi_m,eps_gain,phi_m_ratio,overall\n")

    dataset_files, d_folder, s_folder, s_clf, u_clf, y_label, limit_rows = load_config("config.cfg")

    # Iterating over CSV files in folder
    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not dataset_utils.is_image_dataset(dataset_file):
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")
        else:
            print("Processing Dataset " + dataset_file)
            x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_tabular_dataset(dataset_file, y_label, limit_rows, train_size=0.66)
            x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.5)

            for classifier_string in s_clf:
                # Building and training classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                print("\nBuilding classifier: " + get_classifier_name(classifier))
                start_ms = current_ms()
                if isinstance(classifier, pyod.models.base.BaseDetector):
                    classifier.fit(x_train)
                else:
                    classifier.fit(x_train, y_train)
                train_ms = current_ms()
                y_pred_test = numpy.asarray(label_tags[classifier.predict(x_test)])
                test_time = current_ms() - train_ms
                y_pred_val = numpy.asarray(label_tags[classifier.predict(x_val)])
                print(get_classifier_name(classifier) + " train/test in " + str(train_ms - start_ms) + "/" +
                      str(test_time) + " ms with Accuracy: " +
                      str(sklearn.metrics.accuracy_score(label_tags[y_test], y_pred_test)))

                # Building SPROUT object
                sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
                sprout_obj.load_model(model_tag=MODEL_TAG, clf=classifier,
                                      x_train=x_train, y_train=y_train, label_names=label_tags)

                # Exercising SPROUT
                val_df = sprout_obj.exercise(x=x_val, y=y_val, classifier=classifier)
                val_df.to_csv(os.path.join(OUTPUT_FOLDER, clean_name(dataset_file, d_folder) + "_" +
                                get_classifier_name(classifier) + '_val.csv'), index=False)
                test_df = sprout_obj.exercise(x=x_test, y=y_test, classifier=classifier)
                test_df.to_csv(os.path.join(OUTPUT_FOLDER,  clean_name(dataset_file, d_folder) + "_" +
                               get_classifier_name(classifier) + '_test.csv'), index=False)

                # Experimenting with different UMs
                val_misc = numpy.where(label_tags[y_val] != y_pred_val, 1, 0)
                test_misc = numpy.where(label_tags[y_test] != y_pred_test, 1, 0)
                for um_comb in COMBINATIONS:
                    relevant_features = [UMs[tag] for tag in (um_comb if isinstance(um_comb, list) else [um_comb])]
                    misc_det = DecisionTreeClassifier()
                    misc_det.fit(val_df[relevant_features], val_misc)
                    y_pred_misc = misc_det.predict(test_df[relevant_features])
                    y_pred_clf = numpy.where(y_pred_misc == 0, y_pred_test, 'REJECT')
                    metrics = compute_omission_metrics(label_tags[y_test], y_pred_test, y_pred_clf)

                    with open(OUT_FILE, 'a') as f:
                        f.write(dataset_file + "," + classifier_string + "," + str(um_comb).replace(",", ";") + ",")
                        for dk in metrics:
                            f.write(str(metrics[dk]) + ",")
                        f.write("\n")
