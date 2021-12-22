import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from TrustCalculator import LimeTrust, NativeTrust, EntropyTrust

MY_FILE = "input_folder/NSLKDD_Shuffled.csv"
LABEL_NAME = 'multilabel'


def process_dataset(dataset_name, label_name):
    """
    Method to process an input dataset as CSV
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return:
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")
    print("Dataset loaded: " + str(len(df.index)) + " items")
    y = df[label_name]
    y_bin = np.where(y == "normal", "normal", "attack")

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("Normal data points: " + str(len(normal_frame.index)) + " items ")

    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])

    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_bin, test_size=0.5, shuffle=True)

    # Training/Testing Classifiers
    return x_no_cat, y_bin, x_tr, x_te, y_tr, y_te


if __name__ == '__main__':

    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(MY_FILE, LABEL_NAME)

    # Building Classifier
    classifierName = "XGBoost"
    classifierModel = XGBClassifier()
    classifierModel.fit(X_train, y_train)
    y_pred = classifierModel.predict(X_test)
    y_proba = classifierModel.predict_proba(X_test)

    # Classifier Evaluation
    print(classifierName + " Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))

    # Trust Calculators
    calculators = [
        EntropyTrust(),
        NativeTrust()
    ]

    # Output Dataframe
    xt_numpy = X_test.to_numpy()
    out_df = X_test
    out_df['true_label'] = y_test
    out_df['predicted_label'] = y_pred
    out_df['probabilities'] = y_proba

    for calculator in calculators:

        trust_scores = calculator.trust_scores(xt_numpy, y_proba, classifierModel)
        out_df[calculator.trust_strategy_name()] = trust_scores

    out_df.to_csv('output_folder/out_frame.csv', index=False)

    # explainer = LimeTrust(X_train.to_numpy(), y_train, X_no_cat.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(X_test.to_numpy()))