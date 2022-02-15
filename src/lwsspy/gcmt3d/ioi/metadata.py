import os
import numpy as np


def write_metadata(X, Y, metadir):
    xfile = os.path.join(metadir, "X.npy")
    yfile = os.path.join(metadir, "Y.npy")
    np.save(xfile, X)
    np.save(yfile, Y)


def read_metadata(metadir):
    xfile = os.path.join(metadir, "X.npy")
    yfile = os.path.join(metadir, "Y.npy")
    X = np.load(xfile)
    Y = np.load(yfile)
    return X, Y
