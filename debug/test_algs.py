# Support libs
import os
import random
import time

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from joblib import dump
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Scikit-Learn algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Name of the folder in which look for tabular (CSV) datasets
from sprout.classifiers.Classifier import XGB, UnsupervisedClassifier
from sprout.classifiers.ConfidenceBagging import ConfidenceBagging, ConfidenceBaggingWeighted
from sprout.classifiers.ConfidenceBoosting import ConfidenceBoosting, ConfidenceBoostingWeighted

# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------

CSV_FOLDER = "input_folder/all"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "test_scores.csv"
# Percantage of test data wrt train data
TT_SPLIT = 0.5
# True if debug information needs to be shown
VERBOSE = True
# True if we want to conduct anomaly detection.
# This transforms multi-class labels into binary labels (rule: normal class vs others)
BINARIZE = True

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------


def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)


def get_learners(cont_perc):
    """
    Function to get a learner to use, given its string tag
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    base_learners = [
        XGB(n_estimators=20),
        DecisionTreeClassifier(),
        Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
        #Pipeline([("norm", MinMaxScaler()), ("mnb", MultinomialNB())]),
        #GradientBoostingClassifier(n_estimators=30),
        RandomForestClassifier(n_estimators=20),
        LinearDiscriminantAnalysis(),
        LogisticRegression(),
    ]
    if BINARIZE:
        # If binary classification, we can use unsupervised classifiers also
        cont_alg = cont_perc if cont_perc < 0.5 else 0.5
        base_learners.extend([
            UnsupervisedClassifier(PCA(contamination=cont_alg)),
            UnsupervisedClassifier(HBOS(contamination=cont_alg, n_bins=30)),
            UnsupervisedClassifier(CBLOF(contamination=cont_alg, alpha=0.75, beta=3, n_jobs=-1)),
            UnsupervisedClassifier(IForest(contamination=cont_alg, n_estimators=10, n_jobs=-1)),
            # VotingClassifier(estimators=[('dt', DecisionTreeClassifier()),
            #                              ('pca', UnsupervisedClassifier(PCA(contamination=cont_alg))),
            #                              ('hbos', UnsupervisedClassifier(HBOS(contamination=cont_alg, n_bins=30)))],
            #                  voting='soft'),
            # UnsupervisedClassifier(HBOS(contamination=cont_alg)),
            # GridSearchCV(estimator=ConfidenceBagging(clf=DecisionTreeClassifier()),
            #             param_grid={'n_base': [2, 3, 5], 'max_depth': [2, None]},
            #             scoring='accuracy')
        ])

    learners = []
    for clf in base_learners:
        learners.append(clf)
        for n_base in [5, 10]:
            for s_ratio in [0.2]:
                learners.append(ConfidenceBaggingWeighted(clf=clf, n_base=n_base,
                                                          sampling_ratio=s_ratio, max_features=0.7))
                for n_decisors in [int(n_base / 2)]:
                    learners.append(ConfidenceBagging(clf=clf, n_base=n_base, n_decisors=n_decisors,
                                                      sampling_ratio=s_ratio, max_features=0.7))
            for conf_thr in [0.9]:
                for s_ratio in [0.1, 0.3]:
                    learners.append(ConfidenceBoosting(clf=clf, n_base=n_base,
                                                       learning_rate=2, sampling_ratio=s_ratio,
                                                       contamination=cont_perc, conf_thr=conf_thr))
                    learners.append(ConfidenceBoostingWeighted(clf=clf, n_base=n_base,
                                                       learning_rate=2, sampling_ratio=s_ratio,
                                                       contamination=cont_perc, conf_thr=conf_thr))

    return learners


# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    with open(SCORES_FILE, 'w') as f:
        f.write("dataset_tag,clf,binary,tt_split,acc,misc,mcc,time,model_size\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):

            # if file is a CSV, it is assumed to be a dataset to be processed
            df = pandas.read_csv(full_name, sep=",")
            df = df.sample(frac=1.0)
            if len(df.index) > 100000:
                df = df.iloc[:100000, :]
            if VERBOSE:
                print("\n------------ DATASET INFO -----------------")
                print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
                print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            if VERBOSE:
                print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            if VERBOSE:
                print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            # Binarize if needed (for anomaly detection you need a 2-class problem, requires one of the classes to have NORMAL_TAG)
            normal_perc = None
            y = df[LABEL_NAME].to_numpy()
            if BINARIZE:
                y = numpy.where(df[LABEL_NAME] == "normal", 0, 1)
                if VERBOSE:
                    normal_frame = df.loc[df[LABEL_NAME] == NORMAL_TAG]
                    normal_perc = len(normal_frame.index) / len(df.index)
                    print("Normal data: " + str(len(normal_frame.index)) + " items (" +
                          "{:.3f}".format(100.0 * normal_perc) + "%)")
            elif VERBOSE:
                print("Dataset contains %d Classes" % len(numpy.unique(y)))

            # Set up train test split excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            x_train, x_test, y_train, y_test = ms.train_test_split(x_no_cat, y, test_size=TT_SPLIT, shuffle=True)

            if VERBOSE:
                print('-------------------- CLASSIFIERS -----------------------')

            # Loop for training and testing each learner specified by LEARNER_TAGS
            contamination = 1 - normal_perc if normal_perc is not None else None
            for classifier in get_learners(contamination):

                # Training the algorithm to get a model
                start_time = current_milli_time()
                classifier.fit(x_train, y_train)

                # Quantifying size of the model
                dump(classifier, "clf_dump.bin", compress=9)
                size = os.stat("clf_dump.bin").st_size
                os.remove("clf_dump.bin")

                # Getting Name
                clf_name = classifier.classifier_name() if hasattr(classifier,
                                                                   'classifier_name') else classifier.__class__.__name__
                if clf_name == 'Pipeline':
                    keys = list(classifier.named_steps.keys())
                    clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()

                # Computing metrics
                y_pred = classifier.predict(x_test)
                if hasattr(classifier, 'predict_confidence') and callable(classifier.predict_confidence):
                    y_conf = classifier.predict_confidence(x_test)
                else:
                    y_proba = classifier.predict_proba(x_test)
                    y_conf = numpy.max(y_proba, axis=1)
                conf_ok = y_conf[numpy.where(y_pred == y_test)[0]]
                conf_ok_metrics = [numpy.min(conf_ok), numpy.median(conf_ok), numpy.average(conf_ok),
                                   numpy.max(conf_ok)]
                conf_misc = y_conf[numpy.where(y_pred != y_test)[0]]
                conf_misc_metrics = [numpy.min(conf_misc), numpy.median(conf_misc), numpy.average(conf_misc),
                                     numpy.max(conf_misc)]

                acc = metrics.accuracy_score(y_test, y_pred)
                misc = int((1 - acc) * len(y_test))
                mcc = abs(metrics.matthews_corrcoef(y_test, y_pred))
                if BINARIZE:
                    # Prints metrics for binary classification + train time and model size
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
                    print('%s\t-> TP: %d, TN: %d, FP: %d, FN: %d, Accuracy: %.3f, MCC: %.3f, Conf Diff: %.3f - train time: %d ms'
                          ' - model size: %.3f KB' % (clf_name, tp, tn, fp, fn, acc, mcc, (conf_ok_metrics[2] - conf_misc_metrics[2]),
                                                      current_milli_time() - start_time, size / 1000.0))
                else:
                    # Prints just accuracy for multi-class classification problems, no confusion matrix
                    print('%s\t-> Accuracy: %.3f, Conf Diff: %.3f - train time: %d ms - model size: %.3f KB'
                          % (clf_name, acc, (conf_ok_metrics[2] - conf_misc_metrics[2]),
                             current_milli_time() - start_time, size / 1000.0))

                # Updates CSV file form metrics of experiment
                with open(SCORES_FILE, "a") as myfile:
                    # Prints result of experiment in CSV file
                    myfile.write(full_name + "," + clf_name + "," + str(BINARIZE) + "," +
                                 str(TT_SPLIT) + ',' + str(acc) + "," + str(misc) + "," + str(mcc) + "," +
                                 ";".join(["{:.4f}".format(met) for met in conf_ok_metrics]) + "," +
                                 ";".join(["{:.4f}".format(met) for met in conf_misc_metrics]) + "," +
                                 str(current_milli_time() - start_time) + "," + str(size) + "\n")

                classifier = None
