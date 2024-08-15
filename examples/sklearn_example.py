import sys
sys.path.append('/home/fahad/Project/SPROUT')
from sklearn.ensemble import RandomForestClassifier

from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.dataset_utils import process_binary_tabular_dataset, process_image_dataset
from sprout.utils.general_utils import current_ms
from sprout.utils.sprout_utils import compute_omission_metrics
import sys,os
sys.path.append('/home/fahad/Project/SPROUT/sprout')
MODELS_FOLDER = "../models/"
MODEL_TAG = "sup_all"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for SKLEARN classifiers using NSL-KDD dataset from
    https://www.kaggle.com/datasets/hassan06/nslkdd
    """

    # Reading sample dataset (NSL-KDD)
    x_train, x_test, y_train, y_test, label_names, feature_names = process_image_dataset(dataset_name="MNIST", limit=10000)
    # process_binary_tabular_dataset(dataset_name="input_folder/NSLKDD.csv", label_name="multilabel", limit=10000)


    # Creating classifier clf
    classifier = RandomForestClassifier(n_estimators=10)
    print("\nBuilding classifier: " + get_classifier_name(classifier))
    start_ms = current_ms()
    classifier.fit(x_train, y_train)
    train_ms = current_ms()
    y_pred = classifier.predict(x_test)
    test_time = current_ms() - train_ms
    train_time = train_ms - start_ms

    # Loading SPROUT object with a specific tag amongst those existing
    sprout_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sprout_obj.load_model(model_tag=MODEL_TAG, clf=classifier,
                          x_train=x_train, y_train=y_train, label_names=label_names)
    sprout_df, sprout_pred = sprout_obj.exercise(x=x_test, y=y_test, classifier=classifier)
    # optional for printing all data to CSV file
    # sprout_df.to_csv('sprout_sklearn_df.csv', index=False)

    # Computing metrics and printing results
    metrics = compute_omission_metrics(y_test, sprout_pred, y_pred)

    print("\n\t---------- RESULTS ----------\n"
          "Classifier %s has \n\taccuracy(alpha)=%.3f and \n\tmisclassifications(eps)=%.3f" %
          (get_classifier_name(classifier), metrics['alpha'], metrics['eps']))
    print("Applying the '%s' wrapper gives \n"
          "\tresidual accuracy(alpha_w)=%.3f, \n"
          "\tresidual misclassifications(eps_w)=%.3f, \n"
          "\tomissions(phi)=%.3f,\n"
          "\tcorrect omissions(phim ratio)=%.3f" %
          (MODEL_TAG, metrics['alpha_w'], metrics['eps_w'], metrics['phi'], metrics['phi_m_ratio']))