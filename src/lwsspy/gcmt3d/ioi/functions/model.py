from curses import meta
import os
import numpy as np
from .constants import Constants
from .log import get_iter, get_step


def write_perturbation(perturbation, outdir):
    metadir = os.path.join(outdir, 'meta')
    perturbation = [np.nan if _p is None else _p for _p in perturbation]
    np.save(os.path.join(
        metadir, 'perturbation.npy'), perturbation)


def read_perturbation(outdir):
    metadir = os.path.join(outdir, 'meta')
    perturbation = np.load(os.path.join(metadir, 'perturbation.npy'))
    perturbation = [None if np.isnan(_p) else _p for _p in perturbation]
    return perturbation


def write_scaling(scaling, outdir):
    metadir = os.path.join(outdir, 'meta')
    np.save(os.path.join(
        metadir, 'scaling.npy'), scaling)


def read_scaling(outdir):
    metadir = os.path.join(outdir, 'meta')
    return np.load(os.path.join(metadir, 'scaling.npy')).tolist()


def write_model_names(model_names, outdir):
    metadir = os.path.join(outdir, 'meta')
    model_names = np.save(os.path.join(
        metadir, 'model_names.npy'), np.array(model_names))


def read_model_names(outdir):
    metadir = os.path.join(outdir, 'meta')
    return np.load(os.path.join(metadir, 'model_names.npy')).tolist()


def print_model_names(outdir):

    # Get model names
    model_names = read_model_names(outdir)

    # Print model names
    for _i, _name in model_names:
        print(f"{_i:>5}: {_name}")


def get_simpars(outdir):

    model_names = read_model_names(outdir)
    idx = []
    for _i, _mname in enumerate(model_names):
        if _mname in Constants.nosimpars:
            continue
        else:
            idx.append(_i)

    return idx


def write_model(m, outdir, it, ls=None):
    """Takes in model vector, modldirectory, iteration and linesearch number
    and write model to modl directory.

    Parameters
    ----------
    m : ndarray
        modelvector
    modldir : str
        model directory
    it : int
        iteration number
    ls : int, optional
        linesearch number
    """

    # Create filename that contains both iteration and linesearch number
    if ls is not None:
        fname = f"m_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"m_it{it:05d}.npy"

    file = os.path.join(outdir, 'modl', fname)
    np.save(file, m)


def read_model(outdir, it, ls=None):
    """Reads model vector

    Parameters
    ----------
    modldir : str
        model directory
    it : int
        iteration number
    ls : int, optional
        linesearch number

    Returns
    -------
    ndarray
        model vector
    """
    
    if ls is not None:
        fname = f"m_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"m_it{it:05d}.npy"
    file = os.path.join(outdir, 'modl', fname)
    m = np.load(file)
    return m
