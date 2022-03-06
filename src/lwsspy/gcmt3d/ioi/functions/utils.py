from cgitb import reset
import os
import shutil
import numpy as np
import _pickle as pickle
# from obspy import read_events
from lwsspy.seismo.specfem.read_parfile import read_parfile
from lwsspy.seismo.specfem.write_parfile import write_parfile
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.read_inventory import flex_read_inventory as read_inventory
from lwsspy.seismo.specfem.inv2STATIONS import inv2STATIONS
from lwsspy.seismo.specfem.createsimdir import createsimdir
from lwsspy.utils.io import read_yaml_file, write_yaml_file
from lwsspy.gcmt3d.process_classifier import ProcessParams
from .constants import Constants
from .model import read_model_names, write_model, write_model_names, \
    write_scaling, write_perturbation
from .log import reset_iter, reset_step, write_status
from .get_data import stage_data


def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def createdir(cdir):
    """"Creates directory tree of specified path if it doesn't exist yet

    Parameters
    ----------
    cdir : str
        Path for building directory tree
    """
    if not os.path.exists(cdir):
        os.makedirs(cdir)


def rmdir(cdir):
    """Removes directory tree if it doesnt exist yet

    Parameters
    ----------
    cdir : str
        Removes directory recursively
    """
    shutil.rmtree(cdir)

def downloaddir(inputfile, cmtfilename, get_dirs_only=False):

    # Read inputfile
    input_params = read_yaml_file(inputfile)

    # Get database location
    databasedir = input_params["datadatabase"]

    # Read CMT file
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfilename)
    
    # Get full filename
    outdir = os.path.join(databasedir, cmtsource.eventname)

    # Define the directories
    waveforms = os.path.join(outdir, "waveforms")
    stations = os.path.join(outdir, "stations")

    # Only output outdir if wanted
    if get_dirs_only is False:
        
        # Create maindirectory
        createdir(outdir)

        # WRITESTATUS
        write_status(outdir, "CREATED")

        # Write cmtsolution
        cmtsource.write_CMTSOLUTION_file(os.path.join(outdir, 'init_model.cmt'))

        # Write input file
        write_yaml_file(input_params, os.path.join(outdir, 'input.yml'))

        # Create directories
        createdir(waveforms)
        createdir(stations)

    return outdir, waveforms, stations


# Setup directories
def optimdir(inputfile, cmtfilename, get_dirs_only=False):
    """Sets up source inversion optimization directory

    Parameters
    ----------
    inputfile : str
        location of the input file
    cmtfilename : cmtfilename
        location of original CMTSOLUTION
    get_dirs_only : bool, optional
        Whether to only output the relevant directories, by default False

    Returns
    -------
    _type_
        _description_
    """

    # Read inputfile
    input_params = read_yaml_file(inputfile)

    # Get database location
    databasedir = input_params["database"]

    # Read CMT file
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get full filename
    outdir = os.path.join(databasedir, cmtsource.eventname)

    # Define the directories
    modldir = os.path.join(outdir, "modl")
    metadir = os.path.join(outdir, "meta")
    datadir = os.path.join(outdir, "data")
    simudir = os.path.join(outdir, "simu")
    ssyndir = os.path.join(simudir, "synt")
    sfredir = os.path.join(simudir, "dsdm")
    syntdir = os.path.join(outdir, "synt")
    dsdmdir = os.path.join(outdir, "dsdm")
    costdir = os.path.join(outdir, "cost")
    graddir = os.path.join(outdir, "grad")
    hessdir = os.path.join(outdir, "hess")
    descdir = os.path.join(outdir, "desc")
    optdir = os.path.join(outdir, 'opt')

    # Only output outdir if wanted
    if get_dirs_only is False:

        # Create directories
        createdir(modldir)
        createdir(metadir)
        createdir(datadir)
        createdir(ssyndir)
        createdir(sfredir)
        createdir(syntdir)
        createdir(dsdmdir)
        createdir(costdir)
        createdir(graddir)
        createdir(hessdir)
        createdir(descdir)
        createdir(optdir)

    return outdir, modldir, metadir, datadir, simudir, ssyndir, sfredir, syntdir, \
        dsdmdir, costdir, graddir, hessdir, descdir, optdir


def adapt_processdict(cmtsource, processdict, duration):
    """This is a fairly important method because it implements the
        magnitude dependent processing scheme of the Global CMT project.
        Depending on the magnitude, and depth, the methods chooses which
        wavetypes and passbands are going to be used in the inversion.

    Parameters
    ----------
    cmtsource : lwsspy.seismo.cmtsource.CMTSource
        Earthquake solution
    processdict : dict
        process parameter dictionary
    duration : float
        max duration of the seismograms after processing

    Returns
    -------
    dict
        updated processing parameters
    """

    # Get Process parameters
    PP = ProcessParams(
        cmtsource.moment_magnitude, cmtsource.depth_in_m)
    proc_params = PP.determine_all()

    # Adjust the process dictionary
    for _wave, _process_dict in proc_params.items():

        if _wave in processdict:

            # Adjust weight or drop wave altogether
            if _process_dict['weight'] == 0.0 \
                    or _process_dict['weight'] is None:
                processdict.popitem(_wave)
                continue

            else:
                processdict[_wave]['weight'] = _process_dict["weight"]

            # Adjust pre_filt
            processdict[_wave]['process']['pre_filt'] = \
                [1.0/x for x in _process_dict["filter"]]

            # Adjust trace length depending on the duration
            # given to the class
            processdict[_wave]['process']['relative_endtime'] = \
                _process_dict["relative_endtime"]

            if processdict[_wave]['process']['relative_endtime'] \
                    > duration:
                processdict[_wave]['process']['relative_endtime'] \
                    = duration

            # Adjust windowing config
            for _windict in processdict[_wave]["window"]:
                _windict["config"]["min_period"] = \
                    _process_dict["filter"][3]

                _windict["config"]["max_period"] = \
                    _process_dict["filter"][0]

    # Remove unnecessary wavetypes
    popkeys = []
    for _wave in processdict.keys():
        if _wave not in proc_params:
            popkeys.append(_wave)

    for _key in popkeys:
        processdict.pop(_key, None)

    return processdict


def prepare_inversion_dir(cmtfile, outdir, inputparamfile):

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Read parameterfile
    inputparams = read_yaml_file(inputparamfile)

    # Write the input parameters to the inversion directory (for easy inversion)
    write_yaml_file(inputparams, os.path.join(outdir, 'input.yml'))

    # start label
    start_label = '_' + \
        inputparams['start_label'] if inputparams['start_label'] is not None else ''

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Get initial processing directory
    if inputparams["processparams"] is None:
        processdict = Constants.processdict
    else:
        processdict = read_yaml_file(inputparams['processparams'])

    # Adapting the processing dictionary
    processdict = adapt_processdict(cmtsource, processdict, duration)

    # Writing the new processing file to the directory
    write_yaml_file(processdict, os.path.join(outdir, 'process.yml'))

    # Writing Original CMTSOLUTION
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', cmtsource.eventname + start_label))

    # Write model with generic name for easy access
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', 'init_model.cmt'))


def prepare_model(outdir):

    # Get the initial model
    init_cmt = CMTSource.from_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', 'init_model.cmt'))

    # Read parameterfile
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get the parameters to invert for
    parameters = inputparams['parameters']

    # Get model names
    model_names = list(parameters.keys())

    # Write model names
    write_model_names(model_names, outdir)

    # Get model vector
    model_vector = np.array([getattr(init_cmt, key)
                            for key in parameters.keys()])

    # Write model vector
    write_model(model_vector, outdir, 0, 0)

    # Get scaling
    scaling_vector = np.array([val['scale'] for _, val in parameters.items()])

    # Write scaling vector
    write_scaling(scaling_vector, outdir)

    # Get perturbation
    perturb_vector = np.array([val['pert'] for _, val in parameters.items()])

    # Write scaling vector
    write_perturbation(perturb_vector, outdir)


def prepare_stations(outdir):

    # Read inventory from the station directory and put into a single stations.xml
    inv = read_inventory(os.path.join(outdir, 'meta', 'stations', '*.xml'))

    # Write inventory to a single station directory
    inv.write(os.path.join(outdir, 'meta', 'stations.xml'), format='STATIONXML')

    # Write SPECFEM STATIONS FILE
    inv2STATIONS(inv, os.path.join(outdir, 'meta', 'STATIONS.txt'))


def prepare_simulation_dirs(outdir):

    # Get relevant dirs
    simudir = os.path.join(outdir, 'simu')
    ssyndir = os.path.join(simudir, 'synt')
    sfredir = os.path.join(simudir, 'dsdm')

    # Get input params
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # SPECFEM directory
    specfemdir = inputparams["specfem"]

    # Simulation duration
    simulation_duration = np.round(inputparams["duration"]/60 * 1.02)

    # Get modelparameter names
    model_names = read_model_names(outdir)

    # Stations file
    stations_src = os.path.join(outdir, 'meta', 'STATIONS.txt')

    # Create synthetic directories
    createsimdir(specfemdir, ssyndir,
                 specfem_dict=Constants.specfem_dict)

    # Create one simulation directory for each inversion parameter
    for _i, _mname in enumerate(model_names):

        if _mname in Constants.nosimpars:
            continue
        else:
            # Create
            pardir = os.path.join(sfredir, f"dsdm{_i:05d}")
            createsimdir(specfemdir, pardir,
                         specfem_dict=Constants.specfem_dict)

    # Write stations file for the synthetic directory
    shutil.copyfile(stations_src, os.path.join(ssyndir, "DATA", "STATIONS"))

    # Update Par_file depending on the parameter.
    syn_parfile = os.path.join(ssyndir, "DATA", "Par_file")
    syn_pars = read_parfile(syn_parfile)
    syn_pars["USE_SOURCE_DERIVATIVE"] = False

    # Adapt duration
    syn_pars["RECORD_LENGTH_IN_MINUTES"] = simulation_duration

    # Write Stuff to Par_file
    write_parfile(syn_pars, syn_parfile)

    # Create one simulation directory for each inversion
    for _i, _mname in enumerate(model_names):

        # Half duration an time-shift don't need extra simulations
        if _mname not in Constants.nosimpars:

            pardir = os.path.join(sfredir, f"dsdm{_i:05d}")

            # Write stations file
            # Write stations file for the synthetic directory
            shutil.copyfile(stations_src, os.path.join(
                pardir, "DATA", "STATIONS"))

            # Update Par_file depending on the parameter.
            dsdm_parfile = os.path.join(pardir, "DATA", "Par_file")
            dsdm_pars = read_parfile(dsdm_parfile)

            # Adapt duration
            dsdm_pars["RECORD_LENGTH_IN_MINUTES"] = simulation_duration

            # Check whether parameter is a source location derivative
            if _mname in Constants.locations:
                dsdm_pars["USE_SOURCE_DERIVATIVE"] = True
                dsdm_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = \
                    Constants.source_derivative[_mname]
            else:
                dsdm_pars["USE_SOURCE_DERIVATIVE"] = False

            # Write Stuff to Par_file
            write_parfile(dsdm_pars, dsdm_parfile)

def create_event_dir(cmtfile, inputfile):

    # Get main dir
    out = optimdir(inputfile, cmtfile)
    outdir = out[0]

    # Prep inversion directories
    prepare_inversion_dir(cmtfile, outdir, inputfile)

    # Prepare model
    prepare_model(outdir)


def create_forward_dirs(cmtfile, inputfile):

    # Get main dir
    out = optimdir(inputfile, cmtfile)
    outdir = out[0]

    # Prep inversion directories
    prepare_inversion_dir(cmtfile, outdir, inputfile)

    # Prepare model
    prepare_model(outdir)

    # Get data
    stage_data(outdir)

    # Prep Stations
    prepare_stations(outdir)

    # Preparing the simulation directory
    prepare_simulation_dirs(outdir)

    # Reset iteration counter and linesearch counter
    reset_iter(outdir)
    reset_step(outdir)

    return outdir

def wcreate_forward_dirs(args):
    return create_forward_dirs(*args)


def read_events(eventdir):
    events = []
    for eventfile in os.listdir(eventdir):
        events.append(os.path.join(eventdir, eventfile))
    print(events)
    return events

