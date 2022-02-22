# from lwsspy.utils.reset_cpu_affinity import reset_cpu_affinity
import os
from copy import deepcopy
from obspy import read
from lwsspy.utils.isipython import isipython
from lwsspy.utils.io import read_yaml_file
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.process.process import process_stream
from lwsspy.seismo.process.queue_multiprocess_stream import queue_multiprocess_stream
from lwsspy.seismo.read_inventory import flex_read_inventory as read_inventory
from lwsspy.seismo.stream_multiply import stream_multiply

from .constants import Constants
from .utils import write_pickle
from .model import read_model, read_model_names, read_perturbation


def process_data(outdir):

    metadir = os.path.join(outdir, 'meta')
    datadir = os.path.join(outdir, 'data')

    # Get CMT
    cmtsource = CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Get processing parameters
    processdict = read_yaml_file(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    data = read(os.path.join(datadir, 'waveforms', '*.mseed'))

    # Read metadata
    stations = read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        sdata = deepcopy(data)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        processdict = deepcopy(processdict[_wtype]["process"])

        processdict.pop("relative_starttime")
        processdict.pop("relative_endtime")
        processdict["starttime"] = starttime
        processdict["endtime"] = endtime
        processdict["inventory"] = stations
        processdict.update(dict(
            remove_response_flag=True,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=True)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():
            pdata = process_stream(sdata, **processdict)
        else:
            pdata = queue_multiprocess_stream(
                sdata, processdict, nproc=multiprocesses)

        write_pickle(os.path.join(datadir, f'{_wtype}_processed.pkl'), pdata)


def process_synt(outdir, it, ls):

    # Define directory
    metadir = os.path.join(outdir, 'meta')
    simudir = os.path.join(outdir, 'simu')
    syntdir = os.path.join(outdir, 'synt')

    # Get CMT
    cmtsource = CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Get processing parameters
    processdict = read_yaml_file(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    synt = read(os.path.join(simudir, 'synt', 'OUTPUT_FILES', '*.sac'))

    # Read metadata
    stations = read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        processdict = deepcopy(processdict[_wtype]["process"])

        processdict.pop("relative_starttime")
        processdict.pop("relative_endtime")
        processdict["starttime"] = starttime
        processdict["endtime"] = endtime
        processdict["inventory"] = stations
        processdict.update(dict(
            remove_response_flag=True,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=True)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():
            pdata = process_stream(sdata, **processdict)
        else:
            pdata = queue_multiprocess_stream(
                sdata, processdict, nproc=multiprocesses)

        # Write synthetics
        write_pickle(
            os.path.join(
                syntdir,
                f'{_wtype}_processed_it{it:05d}_ls{ls:05d}.pkl'), pdata)


def wprocess_synt(args):
    process_synt(*args)


def process_dsdm(outdir, nm, it, ls):

    # Define directory
    modldir = os.path.join(outdir, 'modl')
    metadir = os.path.join(outdir, 'meta')
    simudir = os.path.join(outdir, 'simu')
    sfredir = os.path.join(simudir, 'dsdm')
    ssyndir = os.path.join(simudir, 'synt')

    # Get CMT
    cmtsource = CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Read model and model name
    model = read_model(modldir, it, ls)[nm]
    mname = read_model_names(metadir)[nm]
    perturbation = read_perturbation(metadir)[nm]

    # Get processing parameters
    processdict = read_yaml_file(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    if mname in Constants.nosimpars:
        synt = read(os.path.join(ssyndir, 'OUTPUT_FILES', '*.sac'))
    else:
        synt = read(os.path.join(simudir, 'dsdm',
                    f'dsdm{nm:05d}', 'OUTPUT_FILES', '*.sac'))

    # Read metadata
    stations = read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        processdict = deepcopy(processdict[_wtype]["process"])

        processdict.pop("relative_starttime")
        processdict.pop("relative_endtime")
        processdict["starttime"] = starttime
        processdict["endtime"] = endtime
        processdict["inventory"] = stations
        processdict.update(dict(
            remove_response_flag=True,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=True)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():
            pdata = process_stream(sdata, **processdict)
        else:
            pdata = queue_multiprocess_stream(
                sdata, processdict, nproc=multiprocesses)

        if perturbation is not None:
            stream_multiply(pdata, 1.0/perturbation)

        # Compute frechet derivative with respect to time
        if mname == "time_shift":
            pdata.differentiate(method='gradient')
            stream_multiply(pdata, -1.0)
        # If Frechet derivative with respect to depth in m -> divide by 1000
        # since specfem outputs the derivate with respect to depth in km
        elif mname == "depth_in_m":
            stream_multiply(pdata, 1.0/1000.0)

        # Write synthetics
        write_pickle(
            os.path.join(
                sfredir,
                f'dsdm{nm:05d}{_wtype}_processed_it{it:05d}_ls{ls:05d}.pkl'), pdata)


def wprocess_dsdm(args):
    process_dsdm(*args)


def bin():

    from sys import argv

    outdir, metadir, datadir = argv[1:]

    process_data(outdir, metadir, datadir)


if __name__ == "__main__":
    bin()
