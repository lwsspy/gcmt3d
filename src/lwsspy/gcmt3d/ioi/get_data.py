import os
import numpy as np
from lwsspy.utils.io import read_yaml_file
from copy import copy, deepcopy

from .constants import Constants
from .data import write_data, write_data_processed
from .model import write_model
from .metadata import write_metadata
from .gaussian2d import g
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.download_waveforms_to_storage import download_waveforms_to_storage

# %% Get Model CMT


def get_data(outdir: str):

    metadir = os.path.join(outdir, 'meta')
    datadir = os.path.join(outdir, 'data')

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Read input param file
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Define the waveform and station directory and the
    waveformdir = os.path.join(datadir, "waveforms")
    stationdir = os.path.join(metadir, "stations")

    # Download Data Params
    if inputparams["downloadparams"] is None:
        download_dict = Constants.download_dict
    else:
        download_dict = read_yaml_file(inputparams["downloadparams"])

    # Start and End time of the download
    starttime_offset = inputparams["starttime_offset"]
    endtime_offset = inputparams["endtime_offset"]
    starttime = cmtsource.cmt_time + starttime_offset
    endtime = cmtsource.cmt_time + duration + endtime_offset

    download_waveforms_to_storage(
        datadir, starttime=starttime, endtime=endtime,
        waveform_storage=waveformdir, station_storage=stationdir,
        **download_dict)

    return None


def bin():

    import sys
    outdir = sys.argv[1]

    get_data(outdir)
