from sklearn.tree import DecisionTreeClassifier

from sprout.SPROUTObject import exercise_wrapper
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.dataset_utils import load_MNIST

MODELS_FOLDER = "../models/"
MODEL_TAG = "sup_multi"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for sklearn classifiers using MNIST dataset
    To be updated for the thesis
    """

    # Reading sample dataset (MNIST). Loads data as images, flattened for exercising non-DNN classifiers
    x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST(flatten=True, row_limit=5000)

    # Initializes classifier
    clf = DecisionTreeClassifier()

    # Exercises the clf and the SPROUT wrapper
    metrics = exercise_wrapper(MODEL_TAG, MODELS_FOLDER, clf, x_train, y_train, x_test, y_test, label_names, True)

    print("\n\t---------- RESULTS ----------\n"
          "Classifier %s has \n\taccuracy(alpha)=%.3f and \n\tmisclassifications(eps)=%.3f" %
          (get_classifier_name(clf), metrics['alpha'], metrics['eps']))
    print("Applying the '%s' wrapper gives \n"
          "\tresidual accuracy(alpha_w)=%.3f, \n"
          "\tresidual misclassifications(eps_w)=%.3f, \n"
          "\tomissions(phi)=%.3f,\n"
          "\tcorrect omissions(phim ratio)=%.3f" %
          (MODEL_TAG, metrics['alpha_w'], metrics['eps_w'], metrics['phi'], metrics['phi_m_ratio']))
