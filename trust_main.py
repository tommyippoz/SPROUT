import pandas as pd
import sklearn as sk
import numpy as np

from xgboost import XGBClassifier

from TrustCalculator import LimeTrust
from TrustCalculator import EntropyTrust

MY_FILE = "input_folder/NSLKDD_Shuffled.csv"
LABEL_NAME = 'multilabel'



def process_dataset(dataset_name):

    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")
    print("Dataset loaded: " + str(len(df.index)) + " items")

    y = df[LABEL_NAME]
    y_bin = np.where(y == "normal", "normal", "attack")

    # Basic Pre-Processing
    attack_labels = df[LABEL_NAME].unique()
    normal_frame = df.loc[df[LABEL_NAME] == "normal"]
    print("Normal data points: " + str(len(normal_frame.index)) + " items ")

    x = df.drop(columns=[LABEL_NAME])
    x_no_cat = x.select_dtypes(exclude=['object'])

    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_bin, test_size=0.5, shuffle=True)

    # Training/Testing Classifiers
    return x_no_cat, y_bin, x_tr, x_te, y_tr, y_te


if __name__ == '__main__':

    # Reading Dataset
    X, y, X_train, X_test, y_train, y_test = process_dataset(MY_FILE)

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
        EntropyTrust()
    ]

    # Output Dataframe
    xt_numpy = X_test.to_numpy()
    df = X_test
    df['true_label'] = y_test
    df['predicted_label'] = y_pred
    df['probabilities'] = np.array2string(y_proba)

    for calculator in calculators:

        trust_scores = calculator.trust_scores(xt_numpy, y_proba)
        df[calculator.trust_strategy_name()] = trust_scores

    df.to_csv('output_folder/out_frame.csv', index=False)

    # explainer = LimeTrust(X_train.to_numpy(), y_train, X_no_cat.columns, ['normal', 'attack'], classifierModel)
    # print(explainer.trust_scores(X_test.to_numpy()))