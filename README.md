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

Assuming the classifier has such a structure, a QUAIL analysis with three calculators can be set up as follows:

```
# Reading sample dataset (MNIST)
x_train, x_test, y_train, y_test, label_names, feature_names = load_MNIST()

print("Preparing Trust Calculators...")

# Building QUAIL instance and adding Entropy, Bayesian and Neighbour-based Calculators
quail = QuailInstance()
quail.add_calculator_entropy(n_classes=len(label_names))
quail.add_calculator_bayes(x_train=x_train, y_train=y_train, n_classes=len(label_names))
quail.add_calculator_neighbour(x_train=x_train, y_train=y_train, label_names=label_names)

# Building and exercising SKLearn classifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_proba = classifier.predict_proba(x_test)
print("Fit and Prediction completed with Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

# Initializing QUAIL dataset for output
out_df = quail_utils.build_QUAIL_dataset(y_proba, y_pred, y_test, label_names)

# Calculating Trust Measures with QUAIL
q_df = quail.compute_set_trust(data_set=x_test, classifier=classifier)
out_df = pandas.concat([out_df, q_df], axis=1)

# Printing Dataframe
out_df.to_csv('my_quail_df.csv', index=False)
print(out_df.head())
```

Other examples, using ther well-known frameworks for machine learning, can be found in the `examples` folder

## Credits

Developed @ University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
- Leonardo Bargiotti

