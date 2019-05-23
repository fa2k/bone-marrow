import re
import glob
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_simulated_data(data_path):
    """Returns dataset with features and labels (X, y), concatenation of all files, shuffled."""
    Xs, ys = [], []
    for bounds_file in glob.glob(os.path.join(data_path, "*_bmBounds.tab")):
        intensities_file = re.sub(r"_bmBounds\.tab$", "_intensities.tab", bounds_file)
        Xs.append(np.loadtxt(intensities_file))
        ys.append(np.loadtxt(bounds_file))
    return shuffle(np.concatenate(Xs), np.concatenate(ys), random_state=11)


def load_simulated_training_set(data_path):
    x, y = load_simulated_data(data_path)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 11)
    return (xTrain, yTrain)

