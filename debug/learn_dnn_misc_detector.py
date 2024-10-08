import copy
import os
import torch
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn
from PIL import Image
from GenericDataset import *
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from plmodels import *
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from debug.learn_simple_model import sample_data
from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import LogisticReg, get_classifier_name, build_classifier, choose_classifier, \
    UnsupervisedClassifier
from sprout.utils.dataset_utils import process_tabular_dataset, process_image_dataset, is_image_dataset, \
    process_binary_tabular_dataset
from sprout.utils.general_utils import load_config, clean_name, current_ms, clear_folder
from sprout.utils.sprout_utils import build_SPROUT_dataset
from torch.utils.data import DataLoader, Subset

# The folder where to put the new misclassification detector
MODELS_FOLDER = "../models/"
# The folder in which files containing uncertainty measures will be stored
TMP_FOLDER = "tmp"
# The name of the new misclassification detector
MODEL_TAG = "dnn_misc_detector"
# The folder from which image datasets are gonna be loaded
TRAIN_DATA_FOLDER = "/home/fahad/Project/SPROUT/dataset/imagenet/"
# This is to down-sample or over-sample the percentage of misclassified predictions
# in the training set of the misclassification detector
MISC_RATIOS = [None, 0.05, 0.1, 0.2, 0.3]
# Number of Channels
CHANNELS = 0
# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Number of Classes
NUM_CLASSES = 15
#Number of Epochs
MAX_EPOCHS = 100
# -------------------------------------------------------------------------------------------------------
# FUNCTIONS THAT YOU HAVE TO IMPLEMENT

def get_dnn_classifiers():
    """
    This should return a list of classifier objects (Model objects in your code) that you want to use for image class.
    :return: a list of classifier objects
    """
    models = []
    model_name = ['vgg11','densenet121','googlenet','inception_v3','resnet50', 'alexnet']
    for model in model_name:
        model = ImageClassifier(model, num_classes=NUM_CLASSES, learning_rate=1e-3, max_epochs=MAX_EPOCHS)
        models.append(model)
    return models

def get_del_classifiers():
    """
    This should return a classifier objects (Model objects in your code) that you want to use as a checker classifier.
    :return: a classifier object
    """
    model = ImageClassifier('alexnet', num_classes=NUM_CLASSES, learning_rate=1e-3, max_epochs=MAX_EPOCHS)

    return model

def get_list_del_classifiers():
    """
    This should return a classifier objects (Model objects in your code) that you want to use as a checker classifier.
    :return: a classifier object
    """
    models = []
    model_name = ['vgg11','densenet121','googlenet','inception_v3','resnet50', 'alexnet']
    for model in model_name:
        model = ImageClassifier(model_name=model, num_classes=NUM_CLASSES, learning_rate=1e-3, max_epochs=MAX_EPOCHS)
        models.append(model)
    return models
def read_image_dataset(dataset_file):

    """
    This is something that you have to implement
    Has to return 5 items: x_train, y_train, x_test, y_test, labels
    :param dataset_file: the input you need to understand which dataset (and how) you have to read
    :return: x_train, y_train, x_test, y_test, labels
    """
    transform = transforms.Compose([
        transforms.Resize((304, 304)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    custom_data = GenericDatasetLoader(dataset_name=dataset_file, root_dir = TRAIN_DATA_FOLDER, transform= transform, batch_size=8)

    train_loader = custom_data.create_dataloader(split='train')

    test_loader = custom_data.create_dataloader(split='test')

    y_test = custom_data.extract_labels(test_loader)

    global CHANNELS
    CHANNELS = custom_data.get_num_channels(train_loader)

    label_tags = train_loader.dataset.classes
    global NUM_CLASSES
    NUM_CLASSES= len(label_tags)

    # return train_loader, test_loader, X_train, X_test, y_train, y_test, label_tags
    return train_loader, test_loader, y_test, label_tags


def list_image_datasets():
    """
    This should return a list of strings, filenames, folder_names or configuration files that you can use to load datasets
    One item per dataset
    :return:
    """
    dataset_name = ['CIFAR10']
    return dataset_name

# -------------------------------------------------------------------------------------------------------


def compute_datasets_uncertainties():
    """
    Computes uncertainties for all datasets
    :return:
    """

    # Iterates over all datasets
    # You can change this iteration if you have very different ways of getting data from different datasets
    for dataset_file in list_image_datasets():

        if (dataset_file is None) or len(dataset_file) == 0:
            print("Error while processing the dataset")
        else:
            print("Processing Dataset " + dataset_file + "'")

            # Reading Dataset
            # train_loader, test_loader,x_train, x_test, y_train, y_test, label_tags = read_image_dataset(dataset_file)
            train_loader, test_loader, y_test, label_tags = read_image_dataset(dataset_file)


            print("Preparing Uncertainty Calculators...")
            sp_obj = build_supervised_object(train_loader, test_loader, label_tags)

            for classifier in get_dnn_classifiers():
                sprout_obj = copy.deepcopy(sp_obj)
                # Building and exercising classifier
                print(DEVICE)
                classifier.fit(train_dataloader = train_loader)
                # classifier.load_model('/home/fahad/Project/SPROUT/debug/tmp/ResNet_0_model_weights.pth')
                y_proba = classifier.predict_proba(test_loader)
                y_pred = classifier.predict(test_loader)


                # Calculating Trust Measures with SPROUT
                out_df = build_SPROUT_dataset(y_proba, y_pred, y_test, label_tags)
                q_df = sprout_obj.compute_set_trust(data_set = test_loader, classifier=classifier, y_proba=y_proba)
                out_df = pd.concat([out_df, q_df], axis=1)

                # Printing Dataframe containing uncertainty measures
                file_out = os.path.join(TMP_FOLDER, dataset_file + "_" + get_classifier_name(classifier.model) + '.csv')
                if not os.path.exists(os.path.dirname(file_out)):
                    os.mkdir(os.path.dirname(file_out))
                out_df.to_csv(file_out, index=False)
                print("File '" + file_out + "' Printed")


def load_uncertainty_datasets(train_split=0.5, avoid_tags=[], perf_thr=None,
                              label_name="is_misclassification", clean_data=True):
    """
    This loads files containing uncertainty measures for preparing the training/test dataset for misc. detectors
    Theoretically speaking, you should not need to touch it.
    :param train_split: the % of train/test split
    :param avoid_tags: a list of strings, each string is a tag to exclude some files (the tag should appear in the filename of files you want to exclude)
    :param perf_thr: a threshold for the quality of data
    :param label_name: the name of the column of files containing the 0/1 misclassification/correct info (label for the misc. detector)
    :param clean_data:
    :return:
    """
    big_data = []
    for file in os.listdir(TMP_FOLDER):
        if file.endswith(".csv") and (
                (avoid_tags is None) or (len(avoid_tags) == 0) or not any(x in file for x in avoid_tags)):
            df = pandas.read_csv(os.path.join(TMP_FOLDER, file))
            if perf_thr is not None:
                df_acc = 1 - df[label_name].mean()
                if df_acc >= perf_thr:
                    big_data.append(df)
                else:
                    print("File '" + file + "' discarded: has Accuracy " + str(df_acc) + " < " + str(perf_thr))
            else:
                big_data.append(df)
    big_data = pandas.concat(big_data)

    big_data = big_data.sample(frac=1.0)
    big_data = big_data.fillna(0)
    big_data = big_data.replace('null', 0)

    # Cleaning Data
    if clean_data:
        big_data = big_data.drop(columns=["true_label", "predicted_label"])
        big_data = big_data.select_dtypes(exclude=['object'])

    # Creating train-test split
    label = big_data[label_name].to_numpy()
    misc_frac = sum(label) / len(label)
    big_data = big_data.drop(columns=[label_name])
    features = big_data.columns
    big_data = big_data.to_numpy()
    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(big_data, label, train_size=train_split)

    print("Dataset contains " + str(len(label)) + " items and " + str(misc_frac * 100) + "% of misclassifications")

    return x_tr, y_tr, x_te, y_te, features, misc_frac

# Here you have to define your SPROUT object, or rather the uncertainty measures you want to use
# Sume UM are alreay there, you can change them at will
def build_supervised_object(x_train, y_train, label_tags):
    sp_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    classifier = get_list_del_classifiers()
    # if (x_train is not None) and isinstance(x_train, pandas.DataFrame):
    #     x_data = x_train.to_numpy()
    # else:
    #     x_data = x_train
    # # Add UM as much as possible
    # UM1
    # sp_obj.add_calculator_confidence(x_train=x_data, y_train=y_train, confidence_level=0.9)
    # UM2
    sp_obj.add_calculator_maxprob()
    # # # UM3
    sp_obj.add_calculator_entropy(n_classes=len(label_tags))
    # # # UM9
    sp_obj.add_calculator_recloss(x_train=x_train,num_classes=len(label_tags))
    # #
    sp_obj.add_calculator_combined(classifier= classifier[0], x_train=x_train,y_train = y_train, n_classes=len(label_tags))
    sp_obj.add_calculator_multicombined(clf_set=classifier, x_train=x_train, y_train=y_train, n_classes=len(label_tags))
    sp_obj.add_calculator_neighbour(x_train=x_train,y_train=y_train,label_names = label_tags)
    return sp_obj


if __name__ == '__main__':
    """
    Main to calculate trust measures for many datasets using many classifiers.
    Reads preferences from file 'config.cfg'
    """
    # Generating Input data for training Misclassification Predictors
    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)

    # This is to compute the uncertainty measures for image classifiers in each datasets and saving to files
    # The files will then be used at a later stage for creating the train/test set for the misclassification detector
    compute_datasets_uncertainties()
    #
    # sprout_obj =  (None, None, None)
    if os.path.exists(TRAIN_DATA_FOLDER):
        # Merging data into a unique Dataset for training Misclassification Predictors
        # As it is now the train/test split is set to 75-25
        # The perf_thr is there to set a threshold for the quality of data used for learning the misclassification detector
        # The higher the perf_thr, the more data will be discarded (less data, better quality)
        x_train, y_train, x_test, y_test, features, m_frac = \
            load_uncertainty_datasets(train_split=0.75, perf_thr=0.08)

        # Classifiers for Detection (Binary Adjudicators)
        # All of them will be tested and only the best will be chosen as the model for the misc. detector
        m_frac = 0.5 if m_frac > 0.5 else m_frac
        CLASSIFIERS = [GradientBoostingClassifier(n_estimators=30),
                       GradientBoostingClassifier(n_estimators=100),
                       DecisionTreeClassifier(),
                       LinearDiscriminantAnalysis(),
                       RandomForestClassifier(n_estimators=30),
                       RandomForestClassifier(n_estimators=100),
                       GaussianNB(),
                       LogisticReg(),
                       XGBClassifier(n_estimators=30),
                       XGBClassifier(n_estimators=100)
                       ]

        # ------------- Training Binary Adjudicators to Predict Misclassifications -----------

        # Setting up support variables
        best_clf = None
        best_ratio = None
        best_metrics = {"MCC": -10}
        if MISC_RATIOS is None or len(MISC_RATIOS) == 0:
            MISC_RATIOS = [None]

        # Loop for experimenting with all 'CLASSIFIERS'
        for clf_base in CLASSIFIERS:
            clf_name = get_classifier_name(clf_base)

            # Loops over misclassification ratios in the train data for the misclassification detector
            for ratio in MISC_RATIOS:
                clf = copy.deepcopy(clf_base)
                if ratio is not None:
                    x_tr, y_tr = sample_data(x_train, y_train, ratio)
                else:
                    x_tr = x_train
                    y_tr = y_train
                start_ms = current_ms()
                clf.fit(x_tr, y_tr)
                end_ms = current_ms()
                y_pred = clf.predict(x_test)
                mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
                print("[" + clf_name + "][ratio=" + str(ratio) + "] Accuracy: " + str(
                    sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " and MCC of " + str(mcc) + " in " + str((end_ms - start_ms) / 60000) + " mins")
                if mcc > best_metrics["MCC"]:
                    best_ratio = ratio
                    best_clf = clf
                    [tn, fp], [fn, tp] = sklearn.metrics.confusion_matrix(y_test, y_pred)
                    best_metrics = {"MCC": mcc,
                                    "Accuracy": sklearn.metrics.accuracy_score(y_test, y_pred),
                                    "AUC ROC": sklearn.metrics.roc_auc_score(y_test, y_pred),
                                    "Precision": sklearn.metrics.precision_score(y_test, y_pred),
                                    "Recall": sklearn.metrics.recall_score(y_test, y_pred),
                                    "TP": tp,
                                    "TN": tn,
                                    "FP": fp,
                                    "FN": fn}

        print("\nBest classifier is " + get_classifier_name(best_clf) + "/" + str(best_ratio) +
              " with MCC = " + str(best_metrics["MCC"]))

        # Setting up folder to store the SPROUT model
        models_details_folder = MODELS_FOLDER + MODEL_TAG + "/"
        if not os.path.exists(models_details_folder):
            os.mkdir(models_details_folder)
        else:
            clear_folder(models_details_folder)

        # Stores details of the SPROUT object used to build the Binary Adjudicator
        sprout_obj.save_object(models_details_folder)

        # Storing the classifier to be used for Predicting Misclassifications of a Generic Classifier.
        model_file = models_details_folder + "binary_adj_model.joblib"
        joblib.dump(best_clf, model_file, compress=9)

        # Tests if storing was successful
        clf_obj = joblib.load(model_file)
        y_p = clf_obj.predict(x_test)
        if sklearn.metrics.matthews_corrcoef(y_test, y_p) == best_metrics["MCC"]:
            print("Model stored successfully at '" + model_file + "'")
        else:
            print("Error while storing the model - file corrupted")

        # Scores of the SPROUT wrapper
        det_dict = {"analysis tag": MODEL_TAG,
                    "binary classifier": get_classifier_name(best_clf),
                    "train data size": len(y_train),
                    "train data features": numpy.asarray(features),
                    "original misclassification ratio of training set": m_frac,
                    "actual misclassification ratio in training set": best_ratio,
                    "test_mcc": best_metrics["MCC"],
                    "test_acc": best_metrics["Accuracy"],
                    "test_auc": best_metrics["AUC ROC"],
                    "test_p": best_metrics["Precision"],
                    "test_r": best_metrics["Recall"],
                    "test_tp": best_metrics["TP"],
                    "test_tn": best_metrics["TN"],
                    "test_fp": best_metrics["FP"],
                    "test_fn": best_metrics["FN"],
                    }
        with open(models_details_folder + "binary_adjudicator_metrics.txt", 'w') as f:
            for key, value in det_dict.items():
                f.write('%s:%s\n' % (key, value))

        # Plot ROC_AUC
        sklearn.metrics.RocCurveDisplay.from_estimator(best_clf, x_test, y_test)
        plt.savefig(models_details_folder + "binary_adjudicator_aucroc_plot.png")

        if hasattr(best_clf, "feature_importances_"):
            f_imp = dict(zip(numpy.asarray(features), best_clf.feature_importances_))
        elif hasattr(best_clf, "coef_"):
            fi_scores = abs(best_clf.coef_[0])
            f_imp = dict(zip(numpy.asarray(features), fi_scores / sum(fi_scores)))
        else:
            print("No feature importance can be computed for " + get_classifier_name(best_clf))
            f_imp = {}
        with open(models_details_folder + "binary_adjudicator_feature_importances.csv", 'w') as f:
            for key, value in f_imp.items():
                f.write('%s,%s\n' % (key, value))

    else:
            print("Path for the analysis does not exist")
