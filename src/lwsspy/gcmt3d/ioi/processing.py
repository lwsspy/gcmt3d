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

from .utils import write_pickle


def process_data(outdir, metadir, datadir):

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


def bin():

    from sys import argv

    outdir, metadir, datadir = argv[1:]

    process_data(outdir, metadir, datadir)


if __name__ == "__main__":
    bin()
