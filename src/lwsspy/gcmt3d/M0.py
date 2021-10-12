import sys
from pprint import pprint
import numpy as np
from .io import write_fixed_traces
from .measurements import get_all_measurements
from .io import read_measurements
from .io import read_output_traces
from .. import maps as lmap
from .. import seismo as lseis
from ..seismo import stream_multiply
from ..seismo import CMTSource
import os
from typing import Optional
from copy import deepcopy
import _pickle as cPickle


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_ratio(measurement_dict, corr: bool = True, verbose=False):

    if corr:
        val = 'corr_ratio'
    else:
        val = 'dlna'

    ratiodict = dict()

    for _wtype, _wtypedict in measurement_dict.items():

        ratiodict[_wtype] = dict()

        for _comp, _compdict in _wtypedict.items():

            # Get number of element
            n = len(_compdict[val])

            if n == 0:
                ratio = np.nan
            else:
                if val == "dlna":
                    ratio = np.mean(
                        np.sqrt(np.exp(2 * np.array(_compdict[val]))))
                elif val == "corr_ratio":
                    ratio = np.mean(np.array(_compdict[val]))

            # Put into dictionary
            ratiodict[_wtype][_comp] = dict(ratio=ratio, n=n)

            # Log if wanted
            if verbose:
                print(f"{_wtype:7} - {_comp} - R: {ratio:4.2f} - N: {n:d}")

    return ratiodict


def print_ratiodict(ratiodict: dict):

    ratios = []
    for _wtype, _wtypedict in ratiodict.items():
        for _comp in _wtypedict.keys():
            ratios.append(ratiodict[_wtype][_comp])
            print(f"{_wtype:7} {_comp} {ratiodict[_wtype][_comp]:6.4f}")
        print("")

    print(f"Full:   d{np.mean(ratios):6.4f}\n")
    print(f"Actual: {get_factor_from_ratiodict(ratiodict):6.4f}\n")


def get_factor_from_ratiodict(ratiodict, verbose=True):
    """Computes the weighted average of the ratios across components"""

    ratios = []
    nel = 0
    if "mantle" in ratiodict:

        for _comp, _ratdict in ratiodict["mantle"].items():

            # Only use if non-empty
            if ~np.isnan(_ratdict["ratio"]):
                if verbose:
                    print(
                        f"mantle - {_comp} - R: {_ratdict['ratio']:4.2f} - N: {_ratdict['n']:d}")

                nel += _ratdict["n"]
                ratios.append(_ratdict["ratio"] * float(_ratdict["n"]))

        if verbose:
            print(ratios, nel)

        if len(ratios) > 100:
            ratios = np.array(ratios)/float(nel)

        else:

            for _wtype, _compdict in ratiodict.items():
                for _comp, _ratdict in _compdict.items():

                    if ~np.isnan(_ratdict["ratio"]):

                        if verbose:
                            print(
                                f"{_wtype:7} - {_comp} - R: {_ratdict['ratio']:4.2f} - N: {_ratdict['n']:d}")

                        nel += _ratdict["n"]
                        ratios.append(_ratdict["ratio"] * float(_ratdict["n"]))

            if verbose:
                print(ratios, nel)

            if len(ratios) != 0:
                ratios = np.array(ratios)/float(nel)
            else:
                raise ValueError(
                    "Couldn't find proper mean in terms of energy.")

    else:

        for _wtype, _compdict in ratiodict.items():
            for _comp, _ratdict in _compdict.items():

                if ~np.isnan(_ratdict["ratio"]):

                    if verbose:
                        print(
                            f"{_wtype:7} - {_comp} - R: {_ratdict['ratio']:4.2f} - N: {_ratdict['n']:d}")

                    nel += _ratdict["n"]
                    ratios.append(_ratdict["ratio"] * float(_ratdict["n"]))

        if verbose:
            print(ratios, nel)

        if len(ratios) != 0:
            ratios = np.array(ratios)/float(nel)
        else:
            raise ValueError(
                "Couldn't find proper mean in terms of energy.")

    return np.sum(ratios)


def multiply_synt(synt, factor):
    fix_synt = deepcopy(synt)
    for _, _compdict in fix_synt.items():
        for _, _stream in _compdict.items():

            stream_multiply(_stream, factor)
    return fix_synt


def fix_source(event: CMTSource, factor: float) -> CMTSource:

    event = deepcopy(event)
    M0 = event.M0
    event.M0 = factor * M0

    return event


def fix_synthetics(
        cmtdir, label: Optional[str] = None, corr: bool = True,
        verbose=True):

    # Set label
    if label is not None:
        label = "_" + label
    else:
        label = ""

    # Get output traces
    try:
        obsd, synt = read_output_traces(cmtdir)

    except Exception as e:
        if verbose:
            eprint(f'Couldnt read traces for {cmtdir} because {e}.')
        return -1

    # Get event
    try:
        eventfile = os.path.join(cmtdir, os.path.basename(cmtdir) + label)
        event = CMTSource.from_CMTSOLUTION_file(eventfile)
    except Exception as e:
        if verbose:
            eprint(f'Couldnt read event for {cmtdir} because {e}.')
        return -1

    # Measure the traces
    measurementdict_prefix = get_all_measurements(obsd, synt, event)

    # Get factor
    ratiodict = get_ratio(measurementdict_prefix, corr=corr)

    try:
        factor = get_factor_from_ratiodict(ratiodict)
    except Exception as e:
        if verbose:
            eprint(cmtdir, e)
            eprint(ratiodict)
            eprint(measurementdict_prefix)
        return -1

    if verbose:
        print(f"Correction factor: {factor}")

    if np.isnan(factor) or np.isinf(factor):
        if verbose:
            eprint(
                f'Couldnt find good factor for {cmtdir} because factor = {factor}.'
            )
        return -1

    # Fix the traces
    fix_synt = multiply_synt(synt, factor)

    # Fix event
    fix_event = fix_source(event, factor)

    # Measure the traces after fixing
    measurementdict_fix = get_all_measurements(obsd, fix_synt, event)

    # Write the new measurement dictionary
    # Create filename
    filename = f"measurements{label}_fix.pkl"
    outfile = os.path.join(cmtdir, filename)
    if verbose:
        print(f"Outfile: {outfile}")

    # Write to measurement pickle
    with open(outfile, "wb") as f:
        cPickle.dump(measurementdict_fix, f)

    # Write the fixed synthetics to file
    write_fixed_traces(cmtdir, fix_synt)

    # Write fixed cmt solution
    eventout = os.path.join(cmtdir, os.path.basename(cmtdir) + label + "_fix")
    if verbose:
        print(f"Fixed event: {eventout}")
    fix_event.write_CMTSOLUTION_file(eventout)

    return obsd, fix_synt, measurementdict_fix


def fix_database(database: str, label: Optional[str] = None):

    cmts = os.listdir(database)

    for event in cmts:

        # Get file
        cmtdir = os.path.join(database, event)

        # Fix synthetics
        fix_synthetics(cmtdir, label=label, verbose=True)


def bin_fix_event():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='event',
                        help='event directory',
                        type=str)
    parser.add_argument('-l', '--label', dest='label',
                        type=str, default=None, required=False)

    args = parser.parse_args()

    # Fix dlna database
    fix_synthetics(args.event, label=args.label, verbose=True)


def bin_fix_dlna_database():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='database',
                        help='Database directory',
                        type=str)
    parser.add_argument('-l', '--label', dest='label',
                        type=str, default=None, required=False)

    args = parser.parse_args()

    # Fix dlna database
    fix_database(args.database, label=args.label)
