from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sprout.SPROUTObject import exercise_wrapper, SPROUTObject
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.dataset_utils import process_binary_tabular_dataset

MODELS_FOLDER = "../models/"

if __name__ == '__main__':
    """
    Main to calculate confidence measures for SKLEARN classifiers using NSL-KDD dataset from
    https://www.kaggle.com/datasets/hassan06/nslkdd
    """
    # Reading sample dataset (NSLKDD)
    x_train, x_test, x_val, y_val, y_train, y_test, label_names, feature_names = \
        process_binary_tabular_dataset(dataset_name="input_folder/NSLKDD.csv", label_name="multilabel",
                                       limit=150000, test_size=0.35, val_size=0.25)

    # Initializes  classifier
    clf = RandomForestClassifier(n_estimators=10)

    # Builds the SPROUT wrapper
    sp_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    sp_obj.add_calculator_confidence(x_train=x_train, y_train=y_train, confidence_level=0.9)
    sp_obj.add_calculator_maxprob()
    sp_obj.add_calculator_entropy(n_classes=len(label_names) if label_names is not None else 2)
    sp_obj.add_calculator_external(classifier=Pipeline([("norm", MinMaxScaler()), ("clf", MultinomialNB())]),
                                   x_train=x_train, y_train=y_train,
                                   n_classes=len(label_names) if label_names is not None else 2)
    sp_obj.add_calculator_neighbour(x_train=x_train, y_train=y_train, label_names=label_names)
    sp_obj.add_calculator_proximity(x_train=x_train, n_iterations=20, range=0.05)

    # Exercises the clf and the SPROUT wrapper
    metrics = exercise_wrapper(sp_obj, MODELS_FOLDER, clf, x_train, y_train, x_val, y_val, x_test, y_test, label_names, verbose=False)

    print("\n\t---------- RESULTS ----------\n"
          "Classifier %s has \n\taccuracy(alpha)=%.3f and \n\tmisclassifications(eps)=%.3f" %
          (get_classifier_name(clf), metrics['alpha'], metrics['eps']))
    print("Applying the wrapper gives \n"
          "\tresidual accuracy(alpha_w)=%.3f, \n"
          "\tresidual misclassifications(eps_w)=%.3f, \n"
          "\tomissions(phi)=%.3f,\n"
          "\tcorrect omissions(phim ratio)=%.3f" %
          (metrics['alpha_w'], metrics['eps_w'], metrics['phi'], metrics['phi_m_ratio']))
