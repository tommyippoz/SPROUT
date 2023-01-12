# SPROUT - a Safety wraPper thROugh ensembles of UncertainTy measures

Python Framework to improve safety of classifiers by computing quantitative uncertainty in their predictions

## Aim/Concept of the Project

SPROUT implements quantitative uncertainty/confidence measures and integrates well with existing frameworks (e.g., Pandas, Scikit-Learn, PYOD, AutoGluon, and many more) that are commonly used in the machine learning domain for classification. 

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

Assuming the classifier has such a structure, a SPROUT analysis with three calculators can be set up as it can be seen in the `examples` folder

## Credits

Developed @ University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
- Leonardo Bargiotti
