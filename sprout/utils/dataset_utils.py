import os
import shutil
import urllib

import numpy
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets


def load_DIGITS(row_limit=np.nan, as_pandas=False):
    """
    Loads DIGITS dataset from SKLearn
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("DIGITS", limit=row_limit, as_pandas=as_pandas)


def load_MNIST(row_limit=np.nan, as_pandas=False):
    """
    Loads MNIST dataset
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("MNIST", limit=row_limit, as_pandas=as_pandas)


def load_FASHIONMNIST(row_limit=np.nan, as_pandas=False):
    """
    Loads FASHION-MNIST dataset
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("FASHION-MNIST", limit=row_limit, as_pandas=as_pandas)


def process_image_dataset(dataset_name, limit=np.nan, as_pandas=False):
    """
    Gets data for analysis, provided that the dataset is an image dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :param dataset_name: name of the image dataset
    :param limit: specifies if the number of data points has to be cropped somehow (testing purposes)
    :return: many values for analysis
    """
    if dataset_name == "DIGITS":
        mn = datasets.load_digits(as_frame=True)
        feature_list = mn.feature_names
        labels = mn.target_names
        x_digits = mn.data
        y_digits = mn.target
        if (np.isfinite(limit)) & (limit < len(y_digits)):
            x_digits = x_digits[0:limit]
            y_digits = y_digits[0:limit]
        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_digits, y_digits, test_size=0.2, shuffle=True)
        return x_tr, x_te, y_tr, y_te, labels, feature_list

    elif dataset_name == "MNIST":
        mnist_folder = "input_folder/mnist"
        if not os.path.isdir(mnist_folder):
            print("Downloading MNIST ...")
            os.makedirs(mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                          "train-images-idx3-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                          "train-labels-idx1-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                          "t10k-images-idx3-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", mnist_folder)
        return format_mnist(mnist_folder, limit, as_pandas)

    elif dataset_name == "FASHION-MNIST":
        f_mnist_folder = "input_folder/fashion"
        if not os.path.isdir(f_mnist_folder):
            print("Downloading FASHION-MNIST ...")
            os.makedirs(f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                          "train-images-idx3-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                          "train-labels-idx1-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                          "t10k-images-idx3-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", f_mnist_folder)
        return format_mnist(f_mnist_folder, limit, as_pandas)


def download_file(file_url, file_name, folder_name):
    with urllib.request.urlopen(file_url) as response, open(folder_name + "/" + file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def process_tabular_dataset(dataset_name, label_name, limit):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    encoding = pd.factorize(df[label_name])
    y_enc = encoding[0]
    labels = encoding[1]

    # Basic Pre-Processing
    print("Dataset loaded: " + str(len(df.index)) + " items and " + str(len(labels)) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

    return x_tr, x_te, y_tr, y_te, labels, feature_list


def process_binary_tabular_dataset(dataset_name, label_name, limit=numpy.nan):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    print("Dataset loaded: " + str(len(df.index)) + " items")
    y_enc = numpy.where(df[label_name] == "normal", 0, 1)

    # Basic Pre-Processing
    print("Dataset loaded: " + str(len(df.index)) + " items, " + str(sum(y_enc)) + " anomalies")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

    return x_tr, x_te, y_tr, y_te, ["normal", "anomaly"], feature_list


def is_image_dataset(dataset_name):
    """
    Checks if a dataset is an image dataset.
    :param dataset_name: name/path of the dataset
    :return: True if the dataset is not a tabular (CSV) dataset
    """
    return (dataset_name == "DIGITS") or (dataset_name != "MNIST") or (dataset_name != "FASHION-MNIST")


def load_mnist(path, kind='train'):
    """
    Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    :param path: path where the mnist-like dataset is stored
    :param kind: to navigate between mnist-like archives
    :return: train/test set of the mnist-like dataset
    """
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def format_mnist(mnist_folder, limit, as_pandas):
    """
    Loads an mnist-like dataset and provides as output the train/test split plus features
    :param as_pandas: True if output has to be a Pandas Dataframe
    :param mnist_folder: folder to load the mnist-like dataset
    :param limit: specifies if the number of data points has to be cropped somehow (testing purposes)
    :return: many values for analysis
    """
    x_tr, y_tr = load_mnist(mnist_folder, kind='train')
    x_te, y_te = load_mnist(mnist_folder, kind='t10k')

    # Linearizes features in the 28x28 image
    x_tr = np.stack([x.flatten() for x in x_tr])
    x_te = np.stack([x.flatten() for x in x_te])
    x_fmnist = np.concatenate([x_tr, x_te], axis=0)
    y_fmnist = np.concatenate([y_tr, y_te], axis=0)

    # Crops if needed
    if (np.isfinite(limit)) & (limit < len(x_fmnist)):
        x_tr = x_tr[0:int(limit / 2)]
        y_tr = y_tr[0:int(limit / 2)]
        x_te = x_te[0:int(limit / 2)]
        y_te = y_te[0:int(limit / 2)]

    # Lists feature names and labels
    feature_list = ["pixel_" + str(i) for i in np.arange(0, len(x_fmnist[0]), 1)]
    labels = pd.Index(np.unique(y_fmnist), dtype=object)

    if as_pandas:
        return pd.DataFrame(data=x_tr, columns=feature_list),\
               pd.DataFrame(data=x_te, columns=feature_list),\
               pd.DataFrame(data=y_tr, columns=['label']),\
               pd.DataFrame(data=y_te, columns=['label']), labels, feature_list
    else:
        return x_tr, x_te, y_tr, y_te, labels, feature_list
