
"""
Just a few scripts to read outputs of the the inversion

"""

import os
import glob
import _pickle as pickle
from copy import deepcopy
import _pickle as cPickle


def read_traces(wtype, streamdir):
    with open(os.path.join(streamdir, f"{wtype}_stream.pkl"), 'rb') as f:
        d = pickle.load(f)
    return d


def write_fixed_traces(cmtdir: str, fixsynt: dict):

    # Get the output directory
    outputdir = os.path.join(cmtdir, "output")
    syntheticdir = os.path.join(outputdir, "synthetic_fix")

    # Make sure dirs exist
    if os.path.exists(syntheticdir) is False:
        os.makedirs(syntheticdir)

    for _wtype in fixsynt.keys():

        filename = os.path.join(syntheticdir, f"{_wtype}_stream.pkl")
        with open(filename, 'wb') as f:
            cPickle.dump(fixsynt[_wtype]["synt"], f)


def read_output_traces(cmtdir: str, fix: bool = False, verbose: bool = True):
    """Given an Inversion directory, read the output waveforms

    Parameters
    ----------
    cmtdir : str
        Inversion directory
    verbose : str
        Print errors/warnings

    Returns
    -------
    Tuple(dict,dict)
        Contains all wtypes available and the respective components.

    """

    # Get the output directory
    outputdir = os.path.join(cmtdir, "output")
    observeddir = os.path.join(outputdir, "observed")
    syntheticdir = os.path.join(outputdir, "synthetic")
    synthetic_fix_dir = os.path.join(outputdir, "synthetic_fix")

    # Glob all wavetype
    wavedictfiles = glob.glob(os.path.join(observeddir, "*_stream.pkl"))
    wtypes = [os.path.basename(x).split("_")[0] for x in wavedictfiles]

    # Read dictionary
    obsd = dict()
    synt = dict()
    if fix:
        syntfix = dict()

    for _wtype in wtypes:

        try:
            tobsd = read_traces(_wtype, observeddir)
            tsynt = read_traces(_wtype, syntheticdir)
            if fix:
                tsyntf = read_traces(_wtype, synthetic_fix_dir)

            obsd[_wtype] = deepcopy(tobsd)
            synt[_wtype] = dict()
            synt[_wtype]["synt"] = deepcopy(tsynt)

            if fix:
                syntfix[_wtype] = dict()
                syntfix[_wtype]["synt"] = deepcopy(tsyntf)

        except Exception as e:
            if verbose:
                print(f"Couldnt read {_wtype} in {cmtdir} because ")
                print(e)
    if fix:
        return obsd, synt, syntfix
    else:
        return obsd, synt


def read_measurements(cmtdir: str):

    measurement_pickle_before = os.path.join(
        cmtdir, "measurements_before.pkl")
    measurement_pickle_after = os.path.join(
        cmtdir, "measurements_after.pkl")
    try:
        with open(measurement_pickle_before, "rb") as f:
            measurements_before = cPickle.load(f)
        with open(measurement_pickle_after, "rb") as f:
            measurements_after = cPickle.load(f)

        return measurements_before, measurements_after

    except Exception:
        return None


def read_measurements_label(cmtdir: str, label: str):

    measurement_pickle = os.path.join(
        cmtdir, f"measurements_{label}.pkl")

    try:
        with open(measurement_pickle, "rb") as f:
            measurements = cPickle.load(f)

        return measurements

    except Exception:
        return None
