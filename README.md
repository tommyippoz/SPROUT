# QUAIL - QUAntitative confIdence of cLassifiers 

Python Framework to Compute Trust/Confidence in Predictions of Machine Learners

## Aim/Concept of the Project

QUAIL, a Python library for calculating the QUAntitative confIdence of cLassifiers, implements quantitative confidence measures and inte-grates well with existing frameworks (e.g., Pandas, Scikit-Learn) that are commonly used in the machine learning domain. 

While designing, implementing and testing such library we made sure it would work with supervised classifiers, as well as unsupervised classifiers. Also, we created connectors for tabular datasets as well as image datasets such that those classifiers can be fed with different inputs and provide confidence measures related to the execution of many classifiers on datasets with a different structure

## Dependencies

QUAIL needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">SKLearn</a>
- <a href="https://github.com/slundberg/shap">SHAP</a>
- <a href="https://github.com/marcotcr/lime">LIME</a>

## Usage

QUAIL can be piggy-backed after any classifier you may want to use, provided that the classifier implements scikit-learn like interfaces, namely
- classifier.predict(test_set): takes a 2D ndarray and returns an array of predictions for each item of test_set
- classifier.predict_proba(test_set): takes a 2D ndarray and returns a 2D ndarray where each line contains probabilities for a given data point in the test_set

Assuming the classifier has such a structure, a QUAIL analysis can be set up as follows:

## Credits

Developed @ University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
- Leonardo Bargiotti

