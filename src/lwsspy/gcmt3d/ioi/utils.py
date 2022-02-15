import os
import shutil
from lwsspy.seismo.source import CMTSource
from lwsspy.utils.io import read_yaml_file
from lwsspy.gcmt3d.process_classifier import ProcessParams


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
    sfredir = os.path.join(simudir, "frec")
    syntdir = os.path.join(outdir, "synt")
    frecdir = os.path.join(outdir, "frec")
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
        createdir(frecdir)
        createdir(costdir)
        createdir(graddir)
        createdir(hessdir)
        createdir(descdir)
        createdir(optdir)

    return outdir, modldir, metadir, datadir, simudir, ssyndir, sfredir, syntdir, \
        frecdir, costdir, graddir, hessdir, descdir, optdir


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
