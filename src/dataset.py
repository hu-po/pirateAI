import numpy as np
import logging
import random
import pickle
import csv
import json
from matplotlib.image import imread
import matplotlib.pyplot as plt
import src.config as config


def histogram_of_labels(labels, label_dict=config.LABEL_DICT):
    """
    Creates and displays a histogram plot for the given labels
    :param labels: [int] labels
    :param label_dict: {} dictionary of possible labels for human readability
    """
    plt.hist(labels, bins=len(label_dict), normed=True)
    plt.xticks(range(len(label_dict)), label_dict.values())
    plt.ylabel('% of data')
    plt.show()


def plot_images(images, label, pred_label=None, label_dict=config.LABEL_DICT):
    """
    Creates and displays a 3x6 plot of sample images with their labels.
    :param images: [ndarray] input image
    :param label: [ndarray] true label
    :param pred_label: [ndarray] predicted label
    :param label_dict: {} dictionary of possible targets for human readability
    """
    assert len(images) == len(label), "Dimension mismatch between images and labels given"
    fig, axes = plt.subplots(3, 6, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < len(images):  # Less than 9 images
            ax.imshow(images[i], cmap='binary')
            true_label = label_dict[label[i]] if label_dict else label[i]
            if pred_label is None:
                xlabel = "True: %s" % true_label
            else:
                predict_label = label_dict[label[i]] if label_dict else label[i]
                xlabel = "True: %s, Pred: %s" % (true_label, predict_label)
            ax.set_xlabel(xlabel)
            # Remove x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def image_analysis(image, label):
    """
    Displays image and label along with histogram of values
    :param images: [ndarray] input image
    :param label: [ndarray] true label
    """
    assert len(image.shape) == 3, 'Wrong size image given to image_analysis'
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].hist(image.flatten())
    axes[0].set_xlabel('Label %s' % str(label))
    plt.show()


class Dataset(object):
    """
    TThe dataset class provides a way to
    """

    def __init__(self, label_dict=config.LABEL_DICT):
        self.log = logging.getLogger(__name__)
        self.log.info('New dataset, label dictionary: %s' % json.dumps(config.LABEL_DICT))
        self.label_dict = label_dict
        self.input_paths = []
        self.label_list = []

    @classmethod
    def from_datasets(cls, datasets=None, label_dict=config.LABEL_DICT, max_n=None):
        """
        Create dataset from list of other datasets
        :param datasets: [Dataset,] list of datasets
        :param max_n: (int) maximum number of datapoints
        :return: (Dataset) Dataset object
        """
        assert all([isinstance(d, Dataset) for d in datasets]), "Please provide only datasets"
        d = cls(label_dict)
        d.input_paths, d.label_list = d.combine_datasets(datasets)
        if max_n:
            d.input_paths = d.input_paths[:max_n]
            d.label_list = d.label_list[:max_n]
        return d

    @classmethod
    def from_path(cls, dir_path=None, label_dict=config.LABEL_DICT, max_n=None):
        """
        Create dataset from folder
        :param dir_path: (string) path to data folder
        :param max_n: (int) maximum number of datapoints
        :return: (Dataset) Dataset object
        """
        assert dir_path, "Please provide a path to a folder of data"
        d = cls(label_dict)
        d.dir_path = dir_path
        d.input_paths, d.label_list = d._get_data_paths()
        if max_n:
            d.input_paths = d.input_paths[:max_n]
            d.label_list = d.label_list[:max_n]
        return d

    def _get_data_paths(self):
        """
        Puts together path for image and corresponding target.
        :param filepath: directory containing data
        :return:[string],[int] image paths, labels
        """
        assert self.dir_path is not None, "No filepath for the dataset"
        input_paths = []
        targets = []
        # Open CSV file with targets and image filenames
        with open(self.dir_path + '/targets.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                try:
                    input_paths.append(self.dir_path + '/0_' + row[0] + '.png')
                    targets.append(int(row[1]))
                except FileNotFoundError:
                    pass
        return input_paths, targets

    def load_data_sample(self, n=1):
        """
        Random sample of images and targets as np arrays
        :param n: (bool) True to return a sample of 9
        :return:[ndarray],[ndarray] image paths, labels
        """
        if n:  # Get random sample of size n first
            idx = random.sample(range(len(self.input_paths)), n)
            image_paths = [self.input_paths[i] for i in idx]
            targets = [self.label_list[i] for i in idx]
        # Load images and targets as ndarrays
        images = np.asarray([imread(path) for path in image_paths])
        labels = np.asarray(targets)
        return images, labels

    def only_label(self, label=''):
        """
        Returns dataset with only instances of this label
        :param label: (string) label
        :return: {}
        """
        assert label in self.label_dict.values(), "Label not in label dictionary"
        # Get indices for labels corresponding to each label in label_dict
        label_idx = [i for i, x in enumerate(self.label_list) if self.label_dict[x] == label]
        input_paths = [self.input_paths[i] for i in label_idx]
        label_list = [self.label_list[i] for i in label_idx]
        # Create a new dataset
        d = Dataset(label_dict=self.label_dict)
        d.input_paths = input_paths
        d.label_list = label_list
        return d

    @staticmethod
    def combine_datasets(datasets=[]):
        """
        Combine datasets to get list of paths and labels
        :param datasets: [Dataset,] list of datasets
        :return:[string],[int] image paths, labels
        """
        input_paths = []
        targets = []
        for d in datasets:
            input_paths += d.input_paths
            targets += d.label_list
        return input_paths, targets

    def analyze(self):
        """
        Plots some sample images and their label. Histogram of labels.
        """
        print("Size of dataset is %s" % len(self.label_list))
        images, labels = self.load_data_sample(n=18)
        plot_images(images, labels, label_dict=self.label_dict)
        histogram_of_labels(self.label_list, label_dict=self.label_dict)

    def save_to_pickle(self, path=None):
        """
        Saves dataset image and label lists to the given path
        :param path: (string) path to save location
        """
        assert path, "Please provide a path when saving to pickle file"
        with open(path, 'wb') as f:
            pickle.dump((self.input_paths, self.label_list), f)
