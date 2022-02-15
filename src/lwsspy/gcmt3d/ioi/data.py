
import os
import numpy as np


def write_data(data, datadir):
    file = os.path.join(datadir, "data.npy")
    np.save(file, data)


def write_data_processed(data, datadir):
    file = os.path.join(datadir, "data_processed.npy")
    np.save(file, data)


def read_data(datadir):
    return np.load(file=os.path.join(datadir, "data.npy"))


def read_data_processed(datadir):
    return np.load(file=os.path.join(datadir, "data_processed.npy"))
