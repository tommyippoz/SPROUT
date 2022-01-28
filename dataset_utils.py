import os

import numpy as np
import pandas as pd
import sklearn as sk
import wget
from sklearn import datasets


def process_image_dataset(dataset_name, limit):
    """
    Gets data for analysis, provided that the dataset is an image dataset
    :param dataset_name: name of the image dataset
    :param limit: specifies if the number of data points has to be cropped somehow (testing purposes)
    :return: many values for analysis
    """
    if dataset_name == "DIGITS":
        mn = datasets.load_digits(as_frame=True)
        feature_list = mn.columns
        labels = mn.target_names
        x_mnist = mn.frame
        y_mnist = mn.target
        x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_mnist, y_mnist, test_size=0.5, shuffle=True)
        return x_mnist, y_mnist, x_tr, x_te, y_tr, y_te, labels, feature_list

    elif dataset_name == "MNIST":
        mnist_folder = "input_folder/mnist"
        if not os.path.isdir(mnist_folder):
            print("Downloading MNIST ...")
            os.makedirs(mnist_folder)
            wget.download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                          out=mnist_folder)
            wget.download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                          out=mnist_folder)
            wget.download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                          out=mnist_folder)
            wget.download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                          out=mnist_folder)
        return format_mnist(mnist_folder, limit)

    elif dataset_name == "FASHION-MNIST":
        f_mnist_folder = "input_folder/fashion"
        if not os.path.isdir(f_mnist_folder):
            print("Downloading FASHION-MNIST ...")
            os.makedirs(f_mnist_folder)
            wget.download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                          out=f_mnist_folder)
            wget.download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                          out=f_mnist_folder)
            wget.download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                          out=f_mnist_folder)
            wget.download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                          out=f_mnist_folder)
        return format_mnist(f_mnist_folder, limit)


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

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    print("Dataset loaded: " + str(len(df.index)) + " items")
    encoding = pd.factorize(df[label_name])
    y_enc = encoding[0]
    labels = encoding[1]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("Dataset loaded: " + str(len(df.index)) + " items, " + str(len(normal_frame.index)) +
          " normal and " + str(len(labels)) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns
    x_tr, x_te, y_tr, y_te = sk.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

    return x_no_cat, y_enc, x_tr, x_te, y_tr, y_te, labels, feature_list


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


def format_mnist(mnist_folder, limit):
    """
    Loads an mnist-like dataset and provides as output the train/test split plus features
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

    return x_fmnist, y_fmnist, x_tr, x_te, y_tr, y_te, labels, feature_list
