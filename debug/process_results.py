import os

import pandas as pd
import scipy
import sklearn
import sklearn as sk

import utils
from quail.quail_utils import get_classifier_name, get_feature_importance

INPUT_FOLDER = "G:/My Drive/Documents/22-02-14 (SAFECOMP) Trust Score/datasets/raw/"
SCORES_FILENAME = "output_folder/out_rf_detailll.csv"
CORR_FILENAME = "output_folder/corr2.csv"
#RATIOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, None]
RATIOS = [None]
CLASSIFIER_LIST = ["RF"]
#FILTERS = ["NSLKDD", "MNIST", "Baidu", "CIDDS", "RandomForest", "XGBoost", "FastAI", "TabNet", "BackBlaze", None]
FILTERS = [None]


def read_dataset(dataset_name):
    """
    Method to process an input dataset as CSV
    :param dataset_name: name of the file (CSV) containing the dataset
    :return:
    """
    df = pd.read_csv(dataset_name, sep=",")
    df = df.fillna(0)

    return df


def build_dataframe(dataset_filter=None):
    col_labels = []
    df_list = []
    for dataset_file in os.listdir(INPUT_FOLDER):
        if dataset_file.endswith(".csv"):
            if (dataset_filter is None) or (dataset_filter in dataset_file):
                data = read_dataset(INPUT_FOLDER + dataset_file)
                col_labels = data.columns
                data.columns = [i for i in range(0, len(data.columns))]
                df_list.append(data)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df.columns = col_labels

    y = full_df['is_misclassification'].to_numpy()
    full_df = full_df.drop(['true_label', 'predicted_label', 'probabilities'], axis=1)
    x_no_cat = full_df.select_dtypes(exclude=['object'])
    x_no_cat = x_no_cat.drop(["is_misclassification"], axis=1)
    features = x_no_cat.columns

    return full_df, x_no_cat, y, features


def sample_df(df, ratio):
    normal_frame = df.loc[df["is_misclassification"] == 0]
    misc_frame = df.loc[df["is_misclassification"] == 1]

    df_ratio = len(misc_frame.index) / len(normal_frame.index)
    if df_ratio < ratio:
        normal_frame = normal_frame.sample(frac=(df_ratio / (2*ratio)))

    return pd.concat([normal_frame, misc_frame])


def sample_data(x, y, ratio):
    df = pd.DataFrame(x.copy())
    df["is_misclassification"] = y
    df = sample_df(df, ratio)
    df = df.sample(frac=1.0)
    y = df["is_misclassification"].to_numpy()
    df = df.drop(["is_misclassification"], axis=1)
    return df, y


def r_squared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2




def evaluate_corr(df, x, y, features, filter_string):

    corr_array = []

    for feat in features:
        feat_array = []
        feat_values = x[feat].values
        scaled_feat_values = sklearn.preprocessing.MinMaxScaler().fit_transform(feat_values.reshape(-1, 1))
        feat_array.append(r_squared(feat_values, y))
        feat_array.append(abs(scipy.stats.pearsonr(feat_values, y)[0]))
        feat_array.append(abs(scipy.stats.spearmanr(feat_values, y)[0]))
        feat_array.append(1 - scipy.spatial.distance.cosine(feat_values, y))
        feat_array.append(sklearn.feature_selection.chi2(scaled_feat_values, y)[0][0])
        feat_array.append(sklearn.feature_selection.mutual_info_classif(feat_values.reshape(-1, 1), y)[0])
        corr_array.append(feat_array)
        print(filter_string + " / " + feat + " processed.")

    # Write file
    with open(CORR_FILENAME, "a") as myfile:
        myfile.write(str(filter_string) + ",")
        for line in corr_array:
            myfile.write((",". join([str(i) for i in line])) + ",")
        myfile.write("\n")


def evaluate_dataset(df, x, y, features, filter_string):

    for ratio in RATIOS:

        ratio_string = str(ratio) if ratio is not None else "None"
        print("Processing dataset '" + filter_string + "' and ratio " + ratio_string)

        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x, y, test_size=0.33, shuffle=True)

        if ratio is not None:
            x_tr, y_tr = sample_data(x_tr, y_tr, ratio)

        for classifierString in CLASSIFIER_LIST:
            classifierModel = utils.choose_classifier(classifierString, features, "is_misclassification", "recall")
            classifierName = get_classifier_name(classifierModel)

            print("\nTraining classifier: " + classifierName + "\n")

            start_ms = utils.current_ms()
            classifierModel.fit(x_tr, y_tr)
            train_ms = utils.current_ms()
            y_pred = classifierModel.predict(x_te.to_numpy())
            test_time = utils.current_ms() - train_ms
            feat_imp = get_feature_importance(classifierModel)
            # print(feat_imp)

            # Classifier Evaluation
            print(classifierName + " train/test in " + str(train_ms - start_ms) + "/" + str(test_time) + " ms")
            tn, fp, fn, tp = sk.metrics.confusion_matrix(y_te, y_pred).ravel()
            accuracy = sk.metrics.accuracy_score(y_te, y_pred)
            mcc = abs(sk.metrics.matthews_corrcoef(y_te, y_pred))
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
                tp, fn = fn, tp
                tn, fp = fp, tn

            train_att_rate = (y_tr == 1).sum() * 100 / len(y_tr)
            test_att_rate = (y_te == 1).sum() * 100 / len(y_te)

            print("TP: " + str(tp) + ", TN: " + str(tn) + ", FP: " + str(fp) + ", FN: " + str(fn) + ", Accuracy: " +
                  "{:.4f}".format(accuracy) + ", MCC: " + "{:.4f}".format(mcc) + " with " +
                  "{:.2f}".format(train_att_rate) + "/" + "{:.2f}".format(test_att_rate)
                  + "% of misc.")

            # Write file
            with open(SCORES_FILENAME, "a") as myfile:
                # Print DatasetInfo
                myfile.write(filter_string + "," + ratio_string + "," + classifierString + "," +
                             str(len(y_tr)) + "," + str(len(y_te)) + "," +
                             str(train_att_rate) + "," + str(test_att_rate) + ",")
                # Print Scores
                myfile.write(str(tp) + "," + str(tn) + "," + str(fp) + "," + str(fn) + "," + str(accuracy) +
                             "," + str(mcc) + "," + (",". join([str(i) for i in feat_imp])) + "\n")


def evaluate_dataset_detail(df, base_x, y, features, filter_string):

    for my_set in [[11, 12, 13], [1, 8, 11, 12, 13]]:
    #for det in ["Entropy Calculator", "LIMECalculator(20)_11945879 - Sum", "LIMECalculator(20)_11945907 - Sum_Top", "LIMECalculator(20)_11945930 - Intercept", "LIMECalculator(20)_11945951 - Pred", "LIMECalculator(20)_11945971 - Score", "SHAPCalculator(100, 10, bic)_37363130 - Max", "SHAPCalculator(100, 10, bic)_37363131 - Ent", "Trust Calculator on 19 Neighbors_20199 - Trust", "Trust Calculator on 19 Neighbors_20218 - Detail", "External Calculator (NaiveBayes)", "Combined Calculator (XGBoost)", "Multiple Combined Calculator (3 classifiers)", "Multiple Combined Calculator (4 classifiers)", "Confidence Interval (0.9999%)"]:

        ratio_string = str(my_set) if my_set is not None else "None"
        print("Processing dataset '" + filter_string + "' and ratio " + ratio_string)

        x = base_x[[base_x.columns[i] for i in my_set]]

        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x, y, test_size=0.33, shuffle=True)

        for classifierString in CLASSIFIER_LIST:
            classifierModel = utils.choose_classifier(classifierString, features, "is_misclassification", "accuracy")
            classifierName = get_classifier_name(classifierModel)

            print("\nTraining classifier: " + classifierName + "\n")

            start_ms = utils.current_ms()
            classifierModel.fit(x_tr, y_tr)
            train_ms = utils.current_ms()
            y_pred = classifierModel.predict(x_te)
            test_time = utils.current_ms() - train_ms
            feat_imp = get_feature_importance(classifierModel)
            # print(feat_imp)

            # Classifier Evaluation
            print(classifierName + " train/test in " + str(train_ms - start_ms) + "/" + str(test_time) + " ms")
            tn, fp, fn, tp = sk.metrics.confusion_matrix(y_te, y_pred).ravel()
            accuracy = sk.metrics.accuracy_score(y_te, y_pred)
            mcc = abs(sk.metrics.matthews_corrcoef(y_te, y_pred))
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
                tp, fn = fn, tp
                tn, fp = fp, tn

            train_att_rate = (y_tr == 1).sum() * 100 / len(y_tr)
            test_att_rate = (y_te == 1).sum() * 100 / len(y_te)

            print("TP: " + str(tp) + ", TN: " + str(tn) + ", FP: " + str(fp) + ", FN: " + str(fn) + ", Accuracy: " +
                  "{:.4f}".format(accuracy) + ", MCC: " + "{:.4f}".format(mcc) + " with " +
                  "{:.2f}".format(train_att_rate) + "/" + "{:.2f}".format(test_att_rate)
                  + "% of misc.")

            # Write file
            with open(SCORES_FILENAME, "a") as myfile:
                # Print DatasetInfo
                myfile.write(filter_string + "," + ratio_string + "," + classifierString + "," +
                             str(len(y_tr)) + "," + str(len(y_te)) + "," +
                             str(train_att_rate) + "," + str(test_att_rate) + ",")
                # Print Scores
                myfile.write(str(tp) + "," + str(tn) + "," + str(fp) + "," + str(fn) + "," + str(accuracy) +
                             "," + str(mcc) + "," + (",". join([str(i) for i in feat_imp])) + "\n")


if __name__ == '__main__':
    """
    Main to calculate effectiveness of trust measures
    """

    first_time = None

    # Processing Files with Filters
    for dataset_filter in FILTERS:

        df, x, y, features = build_dataframe(dataset_filter=dataset_filter)
        filter_string = str(dataset_filter) if dataset_filter is not None else "None"

        if first_time is None:
            first_time = 1
            # Setup Output File
            with open(SCORES_FILENAME, "w") as myfile:
                # Print Header
                myfile.write("filter,ratio,classifier_name,train_len,test_len,train_misc,test_misc," +
                             "tp,tn,fp,fn,accuracy,mcc," + (",".join(feature.replace(",", ";") for feature in features)) + "\n")
            with open(CORR_FILENAME, "w") as myfile:
                # Print Header
                myfile.write("dataset_name,")
                for feature in features:
                    myfile.write(",".join([feature.replace(",", ";") + "_" + corr for corr in ["R2", "P", "SP", "COS", "CHI", "INFO"]])
                                 + ",")
                myfile.write("\n")

        # evaluate_corr(df, x, y, features, filter_string)
        evaluate_dataset_detail(df, x, y, features, filter_string)


def other():
    # Processing individual Files
    for dataset_file in os.listdir(INPUT_FOLDER):

        if dataset_file.endswith(".csv"):

            data = read_dataset(INPUT_FOLDER + dataset_file)

            y = data['is_misclassification'].to_numpy()
            data = data.drop(['true_label', 'predicted_label', 'probabilities'], axis=1)
            x = data.select_dtypes(exclude=['object'])
            x = x.drop(["is_misclassification"], axis=1)
            features = x.columns

            evaluate_corr(df, x, y, features, dataset_file)
            #evaluate_dataset(df, x, y, features, dataset_file)
