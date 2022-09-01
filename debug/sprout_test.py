import os

import numpy as np
import pandas as pd

from sprout.utils import dataset_utils, sprout_utils
from sprout.utils.general_utils import load_config, choose_classifier, clean_name
from sprout.SPROUTObject import SPROUTObject

if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """

    # Loading Configuration
    dataset_files, classifier_list, y_label, limit_rows = load_config("config.cfg")

    for dataset_file in dataset_files:

        if (not os.path.isfile(dataset_file)) and not dataset_utils.is_image_dataset(dataset_file):

            # Error while Reading Dataset
            print("Dataset '" + str(dataset_file) + "' does not exist / not reachable")

        else:

            print("Processing Dataset " + dataset_file + (" - limit " + str(limit_rows) if np.isfinite(limit_rows) else ""))

            # Reading Dataset
            if dataset_file.endswith('.csv'):
                # Reading Tabular Dataset
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_tabular_dataset(dataset_file, y_label, limit_rows)
            else:
                # Other / Image Dataset
                x_train, x_test, y_train, y_test, label_tags, features = \
                    dataset_utils.process_image_dataset(dataset_file, limit_rows)

            print("Preparing Trust Calculators...")
            quail = SPROUTObject()
            quail.add_calculator_featurebagging(x_train, y_train, n_remove=1)
            quail.add_calculator_montecarlo(x_train, y_train)
            quail.add_calculator_confidence(x_train, y_train, 0.9999)

            for classifier_string in classifier_list:

                # Building and exercising classifier
                classifier = choose_classifier(classifier_string, features, y_label, "accuracy")
                y_proba, y_pred = sprout_utils.build_classifier(classifier, x_train, y_train, x_test, y_test)

                # Initializing SPROUT dataset for output
                out_df = sprout_utils.build_QUAIL_dataset(y_proba, y_pred, y_test, label_tags)

                # Calculating Trust Measures with SPROUT
                q_df = quail.compute_set_trust(data_set=x_test, classifier=classifier)
                out_df = pd.concat([out_df, q_df], axis=1)

                # Printing Dataframe
                file_out = 'output_folder/' + clean_name(dataset_file) + "_" + \
                           sprout_utils.get_classifier_name(classifier) + '.csv'
                out_df.to_csv(file_out, index=False)
                print("File '" + file_out + "' Printed")
