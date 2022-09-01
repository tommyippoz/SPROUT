# SPROUT - Safety wraPper thROugh quantitative UncertainTy 

Python Framework to improve safety of classifiers by computing quantitative uncertainty in their predictions

## Aim/Concept of the Project

SPROUT implements quantitative uncertainty/confidence measures and integrates well with existing frameworks (e.g., Pandas, Scikit-Learn) that are commonly used in the machine learning domain for classification. 

While designing, implementing and testing such library we made sure it would work with supervised classifiers, as well as unsupervised classifiers. Also, we created connectors for tabular datasets as well as image datasets such that those classifiers can be fed with different inputs and provide confidence measures related to the execution of many classifiers on datasets with a different structure

## Dependencies

SPROUT needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">SKLearn</a>

## Usage

SPROUT can wrap any classifier you may want to use, provided that the classifier implements scikit-learn like interfaces, namely
- classifier.predict(test_set): takes a 2D ndarray and returns an array of predictions for each item of test_set
- classifier.predict_proba(test_set): takes a 2D ndarray and returns a 2D ndarray where each line contains probabilities for a given data point in the test_set

Assuming the classifier has such a structure, a SPROUT analysis with three calculators can be set up as follows:

```
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier

from quail import quail_utils
from quail.QuailInstance import QuailInstance
from utils.dataset_utils import load_MNIST

# Reading sample dataset (MNIST)
x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST()

print("Preparing Trust Calculators...")

# Building SPROUT instance and adding Entropy, Bayesian and Neighbour-based Calculators
sp_obj = SproutInstance()
sp_obj.add_calculator_entropy(n_classes=len(label_names))
sp_obj.add_calculator_bayes(x_train=x_train, y_train=y_train, n_classes=len(label_names))
sp_obj.add_calculator_neighbour(x_train=x_train, y_train=y_train, label_names=label_names)

# Building and exercising SKLearn classifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_proba = classifier.predict_proba(x_test)
print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

# Initializing SPROUT dataset for output
out_df = quail_utils.build_SPROUT_dataset(y_proba, y_pred, y_test, label_names)

# Calculating Trust Measures with QUAIL
s_df = sp_obj.compute_set_trust(data_set=x_test, classifier=classifier)
out_df = pandas.concat([out_df, s_df], axis=1)

# Printing Dataframe
out_df.to_csv('my_sprout_df.csv', index=False)
print(out_df.head())
```
Output (run on subpar laptop with limited memory and no GPUs):

![SKLearn_Out](https://github.com/tommyippoz/QUAIL/blob/master/doc/README_SKLearn_Out.png)

Other examples, using ther well-known frameworks for machine learning, can be found in the `examples` folder

## Credits

Developed @ University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
- Leonardo Bargiotti
