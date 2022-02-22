"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# Namespace
import lwsspy.inversion as linv
import lwsspy.plot as lplt
import lwsspy.seismo as lseis
import lwsspy.utils as lutils
import lwsspy.shell as lshell
import lwsspy.geo as lgeo
import lwsspy.math as lmat
from lwsspy.seismo.process.queue_multiprocess_stream import \
    queue_multiprocess_stream
from lwsspy.seismo.window.queue_multiwindow_stream import \
    queue_multiwindow_stream

# Internal
from .process_classifier import ProcessParams
from .measurements import get_all_measurements

# External
import os
import asyncio
import shutil
import datetime
import numpy as np
from copy import deepcopy
from typing import Callable, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from obspy import read, read_events, Stream, Trace
import _pickle as cPickle
import logging

lplt.updaterc(rebuild=False)


# Main parameters
window_dict = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "body.window.yml")

SPECFEM = "/scratch/gpfs/lsawade/MagicScripts/specfem3d_globe"
specfem_dict = {
    "bin": "link",
    "DATA": {
        "Par_file": "file",
    },
    "DATABASES_MPI": "link",
    "OUTPUT_FILES": "dir"
}
invdir = '/home/lsawade/lwsspy/invdir_real'
datadir = os.path.join(invdir, "Data")
scriptdir = os.path.dirname(os.path.abspath(__file__))

# %% Get Model CMT
processdict = lutils.read_yaml_file(os.path.join(scriptdir, "process.yml"))


download_dict = dict(
    network=",".join(['CU', 'G', 'GE', 'IC', 'II', 'IU', 'MN']),
    channel_priorities=["LH*", "BH*"],
)

conda_activation = (
    "source /usr/licensed/anaconda3/2020.7/etc/profile.d/conda.sh && "
    "conda activate lwsspy")
compute_node_login = "lsawade@traverse.princeton.edu"
bash_escape = "source ~/.bash_profile"
parameter_check_list = ['depth_in_m', "time_shift", 'latitude', 'longitude',
                        "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
nosimpars = ["time_shift", "half_duration"]
hypo_pars = ['depth_in_m', "time_shift", 'latitude', 'longitude']
mt_params = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

pardict = dict(
    time_shift=dict(scale=1.0, pert=None),
    depth_in_m=dict(scale=1000.0, pert=None)
)


class GCMT3DInversion:

    # parameter_check_list: list = [
    #     'm_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp',
    #     'latitude', 'longitude', 'depth_in_m', 'time_shift', 'hdur'
    # ]
    parameter_check_list: list = parameter_check_list

    nosimpars: list = nosimpars

    def __init__(
            self,
            cmtsolutionfile: str,
            databasedir: str,
            specfemdir: str,
            processdict: dict = processdict,
            pardict: dict = pardict,
            zero_trace: bool = False,
            # zero_energy: bool = False,
            duration: float = 10800.0,
            starttime_offset: float = -50.0,
            endtime_offset: float = 50.0,
            download_data: bool = True,
            node_login: Optional[str] = None,
            conda_activation: str = conda_activation,
            bash_escape: str = bash_escape,
            download_dict: dict = download_dict,
            damping: float = 0.001,
            hypo_damping: float = 0.001,
            weighting: bool = True,
            normalize: bool = True,
            overwrite: bool = False,
            launch_method: str = "srun -n6 --gpus-per-task=1",
            process_func: Callable = lseis.process_stream,
            window_func: Callable = lseis.window_on_stream,
            multiprocesses: int = 20,
            loglevel: int = logging.DEBUG,
            log2stdout: bool = True,
            log2file: bool = True,
            start_label: Optional[str] = None,
            no_init: bool = False):
        """Main inversion class. It's main input is a CMTSOLUTION file, from
        which an inversion directory is built, given a set of inputs. At this
        point it's mainly built around Specfem, but it is not hard to plug in a 
        different forward modeling technique; see `forward` method.

        Parameters
        ----------
        cmtsolutionfile : str
            As the name suggest the path to a CMTSOLUTION file
        databasedir : str
            Main database directory
        specfemdir : str
            Which specfem directory to link to the inversion
        processdict : dict, optional
            dictionary containing all the processing choices, including 
            the windowing parameters. This is by far the place where most
            choices are made, and surely the place to modify if you change the 
            project away from a global setup, by default processdict
        pardict : dict, optional
            choose which parameters to invert for and how to scale them. Note
            that moment tensor elements are always scaled by the scalar moment, 
            by default pardict
        zero_trace : bool, optional
            if True, the inversion is constrained to have a have zero trace, 
            by default False
        duration : float, optional
            maximum duration of the seismograms used in the inversion, 
            by default 10800.0
        starttime_offset : float, optional
            where to cut of the seismograms in processing, by default -50.0
        endtime_offset : float, optional
            where to cut off the seismograms, by default 50.0
        download_data : bool, optional
            if true the necessary data is going to be downloaded, 
            by default True
        node_login : Optional[str], optional
            If you want to run the entire workflow from a compute node, you must 
            redirect the download through the login node. (not recommended), 
            by default None
        conda_activation : str, optional
            how to activate your conda environment on the login node, 
            by default conda_activation
        bash_escape : str, optional
            E.g., To make conda readily available run the bash profile file,
            by default bash_escape
        download_dict : dict, optional
            Dictionary that defines the download parameters. The default
            dictionary is the one that handles the global cmt case.
            by default download_dict
        damping : float, optional
            value to control the damping of the Gauss-Newton Hessian diagonal, 
            by default 0.001
        hypo_damping : float, optional
            value to control the damping of the Gauss-Newton Hessian diagonal
            elements that correspond to the hypocenter. Only used if damping 
            is set to 0.0, by default 0.001
        weighting : bool, optional
            Use geographical and azimuthal weighting of the misfit function, 
            by default True
        normalize : bool, optional
            normalize the seismograms trace by trace, by default True
        overwrite : bool, optional
            scratch the old inversion directory and overwrite data synthetics 
            etc., by default False
        launch_method : str, optional
            how to launch the simulations, by default "srun -n6 --gpus-per-task=1"
        process_func : Callable, optional
            processing function. Only used if multiprocesses is 1,
            by default lseis.process_stream
        window_func : Callable, optional
            windowing function. Only used if multiprocesses is 1, by default lseis.window_on_stream
        multiprocesses : int, optional
            number of cores used for multiprocessing, by default 20
        loglevel : int, optional
            Choose how much you want to see in log file, by default logging.DEBUG
        log2stdout : bool, optional
            as the name suggests pipes the log to the standard out, 
            by default True
        log2file : bool, optional
            as the name suggests pipes the log to a file, by default True
        start_label : Optional[str], optional
            which label to start from, by default None, in which case the label
            will be set to '_gcmt'
        no_init : bool, optional
            [description], by default False
        """

        # CMTSource
        self.cmtsource = lseis.CMTSource.from_CMTSOLUTION_file(
            cmtsolutionfile)
        self.cmt_out = deepcopy(self.cmtsource)
        self.xml_event = read_events(cmtsolutionfile)[0]

        # File locations
        self.databasedir = os.path.abspath(databasedir)
        self.cmtdir = os.path.join(self.databasedir, self.cmtsource.eventname)

        if start_label is not None:
            start_label = "_" + start_label
        else:
            start_label = "_gcmt"
        self.cmt_in_db = os.path.join(
            self.cmtdir, self.cmtsource.eventname + start_label)
        self.overwrite: bool = overwrite
        self.download_data = download_data

        # Simulation stuff
        self.specfemdir = specfemdir
        self.specfem_dict = specfem_dict
        self.launch_method = launch_method

        # Processing parameters
        self.processdict = processdict
        self.process_func = process_func
        self.window_func = window_func
        self.duration = duration
        self.duration_in_m = np.ceil(duration/60.0)
        self.simulation_duration = np.round(self.duration_in_m * 1.02)
        self.multiprocesses = multiprocesses
        self.sumfunc = lambda results: Stream(results)

        # Inversion dictionary
        self.pardict = pardict

        # Download parameters
        self.starttime_offset = starttime_offset
        self.endtime_offset = endtime_offset
        self.download_dict = download_dict

        # Compute Node does not have internet
        self.conda_activation = conda_activation
        self.node_login = node_login
        self.bash_escape = bash_escape

        # Inversion parameters:
        self.nsim = 1
        self.__get_number_of_forward_simulations__()
        self.not_windowed_yet = True
        self.zero_trace = zero_trace
        self.hessians = []
        self.hessians_scaled = []
        # self.zero_energy = zero_energy
        self.damping = damping
        self.hypo_damping = hypo_damping
        self.normalize = normalize
        self.weighting = weighting
        self.weights_rtz = dict(R=1.0, T=1.0, Z=1.0)
        self.mindepth = 5000.0

        # Initialize data dictionaries
        self.data_dict: dict = dict()
        self.synt_dict: dict = dict()
        self.zero_window_removal_dict: dict = dict()

        # Logging
        self.loglevel = loglevel
        self.log2stdout = log2stdout
        self.log2file = log2file

        # Basic Checks
        self.__basic_check__()

        # Initialize
        self.init()

        # Set iteration number
        self.iteration = 0

    def __basic_check__(self):
        """Checking keys of the parameter dictionary. For supported 

        Raises
        ------
        ValueError
            If parameter is not supported yet.
        ValueError
            Cannot invert for a single moment tensor parameter.
        ValueError
            zerotrace condition requires moment tensor parameters.
        """

        # Check Parameter dict for wrong parameters
        for _par in self.pardict.keys():
            if _par not in self.parameter_check_list:
                raise ValueError(
                    f"{_par} not supported at this point. \n"
                    f"Available parameters are {self.parameter_check_list}")

        # If one moment tensor parameter is given all must be given.
        if any([_par in self.pardict for _par in mt_params]):
            checklist = [_par for _par in mt_params if _par in self.pardict]
            if not all([_par in checklist for _par in mt_params]):
                raise ValueError("If one moment tensor parameter is to be "
                                 "inverted. All must be inverted.\n"
                                 "Update your pardict")
            else:
                self.moment_tensor_inv = True
        else:
            self.moment_tensor_inv = False

        # Check zero trace condition
        if self.zero_trace:
            if self.moment_tensor_inv is False:
                raise ValueError("Can only use Zero Trace condition "
                                 "if inverting for Moment Tensor.\n"
                                 "Update your pardict.")

    def __setup_logger__(self):
        """Setting up logging.
        """

        # create logger
        self.logger = logging.getLogger(f"GCMT3D-{self.cmtsource.eventname}")
        self.logger.setLevel(self.loglevel)
        self.logger.handlers = []

        # stop propagting to root logger
        self.logger.propagate = False

        # create formatter
        formatter = lutils.CustomFormatter()

        # Add file logger if necessary
        if self.log2file:
            fh = logging.FileHandler(self.logfile, mode='w+')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Add stdout logger
        if self.log2stdout:
            sh = logging.StreamHandler()
            sh.setLevel(self.loglevel)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        # Make sure not multiple handlers are created
        self.logger.handler_set = True

        # Starting the log
        lutils.log_bar(
            f"GCMT3D LOG: {self.cmtsource.eventname}",
            plogger=self.logger.info)

    def adapt_processdict(self):
        """This is a fairly important method because it implements the 
        magnitude dependent processing scheme of the Global CMT project.
        Depending on the magnitude, and depth, the methods chooses which
        wavetypes and passbands are going to be used in the inversion."""

        # Logging
        lutils.log_action(
            "Adapting processing dictionary", plogger=self.logger.debug)

        # Get Process parameters
        PP = ProcessParams(
            self.cmtsource.moment_magnitude, self.cmtsource.depth_in_m)
        proc_params = PP.determine_all()

        # Adjust the process dictionary
        for _wave, _process_dict in proc_params.items():

            if _wave in self.processdict:

                # Adjust weight or drop wave altogether
                if _process_dict['weight'] == 0.0 \
                        or _process_dict['weight'] is None:
                    self.processdict.popitem(_wave)
                    continue

                else:
                    self.processdict[_wave]['weight'] = _process_dict["weight"]

                # Adjust pre_filt
                self.processdict[_wave]['process']['pre_filt'] = \
                    [1.0/x for x in _process_dict["filter"]]

                # Adjust trace length depending on the duration
                # given to the class
                self.processdict[_wave]['process']['relative_endtime'] = \
                    _process_dict["relative_endtime"]

                if self.processdict[_wave]['process']['relative_endtime'] \
                        > self.duration:
                    self.processdict[_wave]['process']['relative_endtime'] \
                        = self.duration

                # Adjust windowing config
                for _windict in self.processdict[_wave]["window"]:
                    _windict["config"]["min_period"] = \
                        _process_dict["filter"][3]

                    _windict["config"]["max_period"] = \
                        _process_dict["filter"][0]

        # Remove unnecessary wavetypes
        popkeys = []
        for _wave in self.processdict.keys():
            if _wave not in proc_params:
                popkeys.append(_wave)

        for _key in popkeys:
            self.processdict.pop(_key, None)

        # Dump the processing file in the cmt directory
        lutils.log_action(
            "Writing it to file", plogger=self.logger.debug)
        lutils.write_yaml_file(
            self.processdict, os.path.join(self.cmtdir, "process.yml"))

    def init(self):
        """Initialization of the directories, logger, processing parameters,
        waveform dictionaries, data download, and model vector initialization 
        are run using this function."""

        # Initialize directory
        self.__initialize_dir__()

        # Set up the Logger, so that progress is monitored
        self.__setup_logger__()
        lutils.log_section(
            "Setting up the directories and Waveform dicts",
            plogger=self.logger.info)

        # Fix process dict
        self.adapt_processdict()

        # This has to happen after the processdict is adapted since the
        # process dict is used to create the dictionaries
        self.__initialize_waveform_dictionaries__()

        # Get observed data and process data
        if self.download_data:
            with lutils.Timer(plogger=self.logger.info):
                self.__download_data__()

        # Initialize model vector
        self.__init_model_and_scale__()

    def __initialize_dir__(self):

        # Subdirectories
        self.datadir = os.path.join(self.cmtdir, "data")
        self.waveformdir = os.path.join(self.datadir, "waveforms")
        self.stationdir = os.path.join(self.datadir, "stations")
        self.syntdir = os.path.join(self.cmtdir, "synt")
        self.logfile = os.path.join(
            self.cmtdir, self.cmtsource.eventname + ".log")

        # Create subsynthetic directories
        self.synt_syntdir = os.path.join(self.syntdir, "cmt")
        self.synt_pardirs = dict()
        for _par in self.pardict.keys():
            self.synt_pardirs[_par] = os.path.join(self.syntdir, _par)

        # Create database directory if doesn't exist
        self.__create_dir__(self.databasedir)

        # Create entry directory
        self.__create_dir__(self.cmtdir, overwrite=self.overwrite)

        # Create CMT solution
        if os.path.exists(self.cmt_in_db) is False:
            self.cmtsource.write_CMTSOLUTION_file(self.cmt_in_db)
        else:
            check_cmt = lseis.CMTSource.from_CMTSOLUTION_file(
                self.cmt_in_db)
            if check_cmt != self.cmtsource:
                raise ValueError('Already have a CMTSOLUTION, '
                                 'but it is different from the input one.')

        # Create data directory
        self.__create_dir__(self.datadir)

        # Simulation directory are created as part of the prep simulations
        # routine

    def __init_model_and_scale__(self):

        # Update the scale parameter for the moment tensor inversion
        # depending on the original size of the moment tensor
        if self.moment_tensor_inv:
            for _par, _dict in self.pardict.items():
                if _par in mt_params:
                    _dict["scale"] = self.cmtsource.M0

        # Check whether Mrr, Mtt, Mpp are there for zero trace condition
        # It's important to note here that the zero_trace_array in the
        # following part is simply the gradient of the constraint with repspect
        # to the model parameters. For the zero_energy constraint, we
        # explicitly have have to compute this gradient from measurements,
        # and is therefore "missing" here
        if self.zero_trace:  # and not self.zero_energy:

            self.zero_trace_array = np.array(
                [1.0 if _par in ['m_rr', 'm_tt', 'm_pp'] else 0.0
                 for _par in self.pardict.keys()]
            )
            self.zero_trace_index_array = np.where(
                self.zero_trace_array == 1.)[0]
            self.zero_trace_array = np.append(self.zero_trace_array, 0.0)

        # damping settings
        if self.damping > 0.0:
            # Do nothing as damping is easy to handle
            pass

        elif self.hypo_damping > 0.0:
            # Figure out where to dampen!
            self.hypo_damp_array = np.array([1.0 if _par in hypo_pars else 0.0
                                             for _par in self.pardict.keys()])
            self.hypo_damp_index_array = np.where(
                self.hypo_damp_array == 1.)[0]

            self.logger.debug("Hypocenter-Damping Indeces:")
            self.logger.debug(self.hypo_damp_index_array)

        # Get the model vector given the parameters to invert for
        self.model = np.array(
            [getattr(self.cmtsource, _par) for _par in self.pardict.keys()])
        self.model_idxdict = {_par: _i for _i,
                              _par in enumerate(self.pardict.keys())}
        self.init_model = 1.0 * self.model
        self.pars = [_par for _par in self.pardict.keys()]

        # Create scaling vector
        # self.scale = np.array(
        #     [10**lmat.magnitude(getattr(self.cmtsource, _par))
        #      if _par not in mt_params else _dict['scale']
        #      for _par, _dict in self.pardict.items()])
        self.scale = np.array([_dict['scale']
                              for _, _dict in self.pardict.items()])

        self.scaled_model = self.model/self.scale
        self.init_scaled_model = 1.0 * self.scaled_model

    def __initialize_waveform_dictionaries__(self):

        for _wtype in self.processdict.keys():
            self.data_dict[_wtype] = Stream()
            self.synt_dict[_wtype] = dict()
            self.synt_dict[_wtype]["synt"] = Stream()

            for _par in self.pardict.keys():
                self.synt_dict[_wtype][_par] = Stream()

    def process_data(self):
        lutils.log_section(
            "Loading and processing the data",
            plogger=self.logger.info)

        with lutils.Timer(plogger=self.logger.info):
            self.__load_data__()
        with lutils.Timer(plogger=self.logger.info):
            self.__process_data__()

    def process_synt(self):
        lutils.log_section(
            "Loading and processing the modeled data",
            plogger=self.logger.info)

        with lutils.Timer(plogger=self.logger.info):
            self.__load_synt__()
        with lutils.Timer(plogger=self.logger.info):
            self.__process_synt__()

    def get_windows(self):
        """Runs all necessary functions to compute windows of similarity 
        between synthetics and observed data."""

        self.__prep_simulations__()
        self.__write_sources__()

        # Run first set of simulations
        with lutils.Timer(plogger=self.logger.info):
            self.__run_simulations__()
        self.process_all_synt()

        # Copy the initial synthetics
        self.copy_init_synt()

        # Window the data
        with lutils.Timer(plogger=self.logger.info):
            self.__window__()

        # Prep next set of simulations
        with lutils.Timer(plogger=self.logger.info):
            self.__prep_simulations__()

        self.not_windowed_yet = False

    def copy_init_synt(self):
        """Just copies the initial synthetics so that we can make measurements 
        plots and measurements after inversion."""

        # Copy the initial waveform dictionary
        self.synt_dict_init = deepcopy(self.synt_dict)

    def __compute_weights__(self):
        """Computing the geographical and azimuthal weights."""

        # Computing the weights
        lutils.log_bar("Computing Weights", plogger=self.logger.info)

        # Weight dictionary
        self.weights = dict()
        self.weights["event"] = [
            self.cmtsource.latitude, self.cmtsource.longitude]

        waveweightdict = dict()
        for _i, (_wtype, _stream) in enumerate(self.data_dict.items()):

            # Dictionary to keep track of the sum in each wave type.
            waveweightdict[_wtype] = 0

            # Get wave type weight from process.yml
            self.weights[_wtype] = dict()
            waveweight = self.processdict[_wtype]["weight"]
            self.weights[_wtype]["weight"] = deepcopy(waveweight)

            # Create dict to access traces
            RTZ_traces = dict()
            for _component, _cweight in self.weights_rtz.items():

                # Copy compnent weight to dictionary
                self.weights[_wtype][_component] = dict()
                self.weights[_wtype][_component]["weight"] = deepcopy(_cweight)

                # Create reference
                RTZ_traces[_component] = []

                # Only add ttraces that have windows.
                for _tr in _stream:
                    if _tr.stats.component == _component \
                            and len(_tr.stats.windows) > 0:
                        RTZ_traces[_component].append(_tr)

                # Get locations
                latitudes = []
                longitudes = []
                for _tr in RTZ_traces[_component]:
                    latitudes.append(_tr.stats.latitude)
                    longitudes.append(_tr.stats.longitude)
                latitudes = np.array(latitudes)
                longitudes = np.array(longitudes)

                # Save locations into dict
                self.weights[_wtype][_component]["lat"] = deepcopy(latitudes)
                self.weights[_wtype][_component]["lon"] = deepcopy(longitudes)

                # Get azimuthal weights for the traces of each component
                if len(latitudes) > 1 and len(longitudes) > 2:
                    azi_weights = lgeo.azi_weights(
                        self.cmtsource.latitude,
                        self.cmtsource.longitude,
                        latitudes, longitudes, nbins=12, p=0.5)

                    # Save azi weights into dict
                    self.weights[_wtype][_component]["azimuthal"] \
                        = deepcopy(azi_weights)

                    # Get Geographical weights
                    gw = lgeo.GeoWeights(latitudes, longitudes)
                    _, _, ref, _ = gw.get_condition()
                    geo_weights = gw.get_weights(ref)

                    # Save geo weights into dict
                    self.weights[_wtype][_component]["geographical"] \
                        = deepcopy(geo_weights)

                    # Compute Combination weights.
                    weights = (azi_weights * geo_weights)
                    weights /= np.sum(weights)/len(weights)
                    self.weights[_wtype][_component]["combination"] = deepcopy(
                        weights)

                # Figuring out weighting for 2 events does not make sense
                # There is no relative clustering.
                elif len(latitudes) == 2 and len(longitudes) == 2:
                    self.weights[_wtype][_component]["azimuthal"] = [0.5, 0.5]
                    self.weights[_wtype][_component]["geographical"] = [
                        0.5, 0.5]
                    self.weights[_wtype][_component]["combination"] = [
                        0.5, 0.5]
                    weights = [0.5, 0.5]

                elif len(latitudes) == 1 and len(longitudes) == 1:
                    self.weights[_wtype][_component]["azimuthal"] = [1.0]
                    self.weights[_wtype][_component]["geographical"] = [1.0]
                    self.weights[_wtype][_component]["combination"] = [1.0]
                    weights = [1.0]
                else:
                    self.weights[_wtype][_component]["azimuthal"] = []
                    self.weights[_wtype][_component]["geographical"] = []
                    self.weights[_wtype][_component]["combination"] = []
                    weights = []

                # Add weights to traces
                for _tr, _weight in zip(RTZ_traces[_component], weights):
                    _tr.stats.weights = _cweight * _weight
                    waveweightdict[_wtype] += np.sum(_cweight * _weight)

        # Normalize by component and aximuthal weights
        for _i, (_wtype, _stream) in enumerate(self.data_dict.items()):
            # Create dict to access traces
            RTZ_traces = dict()

            for _component, _cweight in self.weights_rtz.items():
                RTZ_traces[_component] = []
                for _tr in _stream:
                    if _tr.stats.component == _component \
                            and "weights" in _tr.stats:
                        RTZ_traces[_component].append(_tr)

                self.weights[_wtype][_component]["final"] = []
                for _tr in RTZ_traces[_component]:
                    _tr.stats.weights /= waveweightdict[_wtype]

                    self.weights[_wtype][_component]["final"].append(
                        deepcopy(_tr.stats.weights))

        with open(os.path.join(self.cmtdir, "weights.pkl"), "wb") as f:
            cPickle.dump(deepcopy(self.weights), f)

    def process_all_synt(self):
        """Runs all processing function on synthetic seismograms and 
        corresponding Frechet derivatives."""

        # Logging
        lutils.log_section(
            "Loading and processing all modeled data",
            plogger=self.logger.info)

        with lutils.Timer(plogger=self.logger.info):
            self.__load_synt__()
            self.__load_synt_par__()
            # self.__remove_zero_windows_on_synt__()

        with lutils.Timer(plogger=self.logger.info):
            self.__process_synt__()
            self.__process_synt_par__()

    def __get_number_of_forward_simulations__(self):
        """Computes the number of necessary forward simulations. E.g. we don't
        need a simulation for the centroid time_shift."""

        # For normal forward synthetics
        self.nsim = 1

        # Add one for each parameters that requires a forward simulation
        for _par in self.pardict.keys():
            if _par not in self.nosimpars:
                self.nsim += 1

    def __download_data__(self):
        """Uses the download dictionary parameters to download the data 
        necessary for the inversion"""

        # Setup download times depending on input...
        # Maybe get from process dict?
        starttime = self.cmtsource.cmt_time + self.starttime_offset
        endtime = self.cmtsource.cmt_time + self.duration \
            + self.endtime_offset

        lutils.log_bar("Data Download", plogger=self.logger.info)

        if self.node_login is None:
            lseis.download_waveforms_to_storage(
                self.datadir, starttime=starttime, endtime=endtime,
                **self.download_dict)

        else:
            from subprocess import Popen, PIPE
            download_cmd = (
                f"download-data "
                f"-d {self.datadir} "
                f"-s {starttime} "
                f"-e {endtime} "
                f"-N {self.download_dict['network']} "
                f"-C {self.download_dict['channel']} "
                f"-L {self.download_dict['location']}"
            )

            login_cmd = ["ssh", "-T", self.node_login]
            comcmd = f"""
            {self.conda_activation}
            {download_cmd}
            """

            lutils.log_action(
                f"Logging into {' '.join(login_cmd)} and downloading",
                plogger=self.logger.info)
            self.logger.debug(f"Command: \n{comcmd}\n")

            with Popen(["ssh", "-T", self.node_login],
                       stdin=PIPE, stdout=PIPE, stderr=PIPE,
                       universal_newlines=True) as p:
                output, error = p.communicate(comcmd)

            if p.returncode != 0:
                self.logger.error(output)
                self.logger.error(error)
                self.logger.error(p.returncode)
                raise ValueError("Download not successful.")

    def __load_data__(self):
        """Loads the data into the prepared waveform dictionaries one for each
        wavetype.
        """

        lutils.log_action("Loading the data", plogger=self.logger.info)

        # Load Station data
        self.stations = lseis.read_inventory(
            os.path.join(self.stationdir, "*.xml"))

        # Load seismic data
        self.data = read(os.path.join(self.waveformdir, "*.mseed"))
        self.raw_data = self.data.copy()

        # Populate the data dictionary.
        for _wtype, _stream in self.data_dict.items():
            self.data_dict[_wtype] = self.data.copy()

    def __process_data__(self):
        """Function that runs the processing function on the data using 
        the wavetype specific processing parameters. """

        # Process each wavetype.
        for _wtype, _stream in self.data_dict.items():
            lutils.log_action(
                f"Processing data for {_wtype}",
                plogger=self.logger.info)

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])

            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict["inventory"] = self.stations
            processdict.update(dict(
                remove_response_flag=True,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude,
                geodata=True)
            )

            if self.multiprocesses < 1:
                self.data_dict[_wtype] = self.process_func(
                    _stream, **processdict)
            else:
                lutils.log_action(
                    f"Parallel processing using {self.multiprocesses} cores",
                    plogger=self.logger.debug)
                self.data_dict[_wtype] = queue_multiprocess_stream(
                    _stream, processdict, nproc=self.multiprocesses)

    def __load_synt__(self):
        """Loads the synthetics into the prepared waveform dictionaries one for
        each wavetype. 
        """

        # if self.specfemdir is not None:
        # Load forward data
        lutils.log_action("Loading forward synthetics",
                          plogger=self.logger.info)
        temp_synt = read(os.path.join(
            self.synt_syntdir, "OUTPUT_FILES", "*.sac"))

        for _wtype in self.processdict.keys():
            self.synt_dict[_wtype]["synt"] = temp_synt.copy()

    def __load_synt_par__(self):
        """Loads the frechet derivatives into the prepared waveform dictionaries
        one for each wavetype. 
        """

        # Load frechet data
        lutils.log_action("Loading parameter synthetics",
                          plogger=self.logger.info)
        for _par, _pardirs in self.synt_pardirs.items():
            lutils.log_action(f"    {_par}", plogger=self.logger.info)

            if _par in self.nosimpars:
                temp_synt = read(os.path.join(
                    self.synt_syntdir, "OUTPUT_FILES", "*.sac"))
            else:
                # Load foward/perturbed data
                temp_synt = read(os.path.join(
                    _pardirs, "OUTPUT_FILES", "*.sac"))

            # Populate the wavetype Streams.
            for _wtype, _ in self.data_dict.items():
                self.synt_dict[_wtype][_par] = temp_synt.copy()

        del temp_synt

    def __process_synt__(self, no_grad=False):
        """Process the synthetics using the processing functions and
        parameters."""

        if self.multiprocesses > 1:
            parallel = True
            # p = mpp.Pool(processes=self.multiprocesses)
            lutils.log_action(
                f"Processing in parallel using {self.multiprocesses} cores",
                plogger=self.logger.debug)
        else:
            parallel = False

        for _wtype in self.processdict.keys():

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict["inventory"] = self.stations
            processdict.update(dict(
                remove_response_flag=False,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude)
            )
            lutils.log_action(
                f"Processing {_wtype}/synt: "
                f"{len(self.synt_dict[_wtype]['synt'])} waveforms",
                plogger=self.logger.info)

            if parallel:
                self.synt_dict[_wtype]["synt"] = queue_multiprocess_stream(
                    self.synt_dict[_wtype]["synt"], processdict,
                    nproc=self.multiprocesses)
            else:
                self.synt_dict[_wtype]["synt"] = self.process_func(
                    self.synt_dict[_wtype]["synt"], self.stations,
                    **processdict)

        if parallel:
            pass
            # p.close()

    def __process_synt_par__(self):
        """Process the frechet derivatives using the processing functions and
        parameters."""

        if self.multiprocesses > 1:
            parallel = True
            # p = mpp.Pool(processes=self.multiprocesses)
            lutils.log_action(
                f"Processing in parallel using {self.multiprocesses} cores",
                plogger=self.logger.debug)
        else:
            parallel = False

        for _wtype in self.processdict.keys():

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict["inventory"] = self.stations
            processdict.update(dict(
                remove_response_flag=False,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude)
            )

            # Process each wavetype.
            for _par, _parsubdict in self.pardict.items():
                lutils.log_action(
                    f"Processing {_wtype}/{_par}: "
                    f"{len(self.synt_dict[_wtype][_par])} waveforms",
                    plogger=self.logger.info)

                if _par in self.nosimpars:
                    self.synt_dict[_wtype][_par] = \
                        self.synt_dict[_wtype]["synt"].copy()

                else:
                    if parallel:
                        self.synt_dict[_wtype][_par] = \
                            queue_multiprocess_stream(
                            self.synt_dict[_wtype][_par], processdict,
                            nproc=self.multiprocesses)
                    else:
                        self.synt_dict[_wtype][_par] = self.process_func(
                            self.synt_dict[_wtype][_par], self.stations,
                            **processdict)
                    # divide by perturbation value and scale by scale length
                if _parsubdict["pert"] is not None:
                    if _parsubdict["pert"] != 1.0:
                        lseis.stream_multiply(
                            self.synt_dict[_wtype][_par],
                            1.0/_parsubdict["pert"])

                # Compute frechet derivative with respect to time
                if _par == "time_shift":
                    self.synt_dict[_wtype][_par].differentiate(
                        method='gradient')
                    lseis.stream_multiply(self.synt_dict[_wtype][_par], -1.0)
                if _par == "depth_in_m":
                    lseis.stream_multiply(
                        self.synt_dict[_wtype][_par], 1.0/1000.0)

        if parallel:
            pass
            # p.close()

    def __window__(self):
        """If both synthetics and observed data have been processed, they can
        be window according to their similarity. This function handles the
        computation of windows according to wavetype and windowing parameters
        in the wave type dictionary."""

        # Debug flag
        debug = True if self.loglevel >= 20 else False

        for _wtype in self.processdict.keys():
            lutils.log_action(
                f"Windowing {_wtype}", plogger=self.logger.info)

            for window_dict in self.processdict[_wtype]["window"]:

                # Wrap window dictionary
                wrapwindowdict = dict(
                    station=self.stations,
                    event=self.xml_event,
                    config_dict=window_dict,
                    _verbose=debug
                )

                # Serial or Multiprocessing
                if self.multiprocesses <= 1:
                    self.window_func(
                        self.data_dict[_wtype],
                        self.synt_dict[_wtype]["synt"],
                        **wrapwindowdict)
                else:
                    self.data_dict[_wtype] = queue_multiwindow_stream(
                        self.data_dict[_wtype],
                        self.synt_dict[_wtype]["synt"],
                        wrapwindowdict, nproc=self.multiprocesses)

            if len(self.processdict[_wtype]["window"]) > 1:
                lutils.log_action(
                    f"Merging {_wtype}windows", plogger=self.logger.info)
                self.merge_windows(
                    self.data_dict[_wtype],
                    self.synt_dict[_wtype]["synt"])

            # After each trace has windows attached continue
            lseis.add_tapers(self.data_dict[_wtype], taper_type="tukey",
                             alpha=0.25, verbose=debug)

            # Some traces aren't even iterated over..
            for _tr in self.data_dict[_wtype]:
                if "windows" not in _tr.stats:
                    _tr.stats.windows = []

    def merge_windows(self, data_stream: Stream, synt_stream: Stream):
        """After windowing, the windows are often directly adjacent. In such
        cases, we can simply unite the windows. The `merge_windows` method 
        calls the appropriate functions to handle that."""

        for obs_tr in data_stream:
            try:
                synt_tr = synt_stream.select(
                    station=obs_tr.stats.station,
                    network=obs_tr.stats.network,
                    component=obs_tr.stats.component)[0]
            except Exception as e:
                self.logger.warning(e)
                self.logger.warning(
                    "Couldn't find corresponding synt for "
                    f"obsd trace({obs_tr.id}): {e}")
                continue
            if len(obs_tr.stats.windows) > 1:
                obs_tr.stats.windows = lseis.merge_trace_windows(
                    obs_tr, synt_tr)

    def optimize(self, optim: linv.Optimization):
        """The main driver of the inversion process. Given an optimization 
        structure this function will optimize the moment tensor. 
        It's important to note that the optimization struct is agnostic to the 
        problem. It only cares about cost, gradient, and hessian that are
        provided by the compute_cost_gradient_hessian function, given a 
        certain model vector."""

        try:
            if self.zero_trace:
                model = np.append(deepcopy(self.scaled_model), 1.0)
            else:
                model = deepcopy(self.scaled_model)
            optim_out = deepcopy(optim.solve(optim, model))
            self.model = deepcopy(optim_out.model)
            return optim_out
        except Exception as e:
            print(e)
            return optim

    def __prep_simulations__(self):
        """This function prepares the parameter files for the 
        forward simulations depending on stations, frechet derivatives etc."""

        lutils.log_action("Prepping simulations", plogger=self.logger.info)
        # Create forward directory
        if self.specfemdir is not None:
            lseis.createsimdir(self.specfemdir, self.synt_syntdir,
                               specfem_dict=self.specfem_dict)
        else:
            self.__create_dir__(self.syntdir)

        # Create one directory synthetics and each parameter
        for _par, _pardir in self.synt_pardirs.items():
            if _par not in self.nosimpars:
                if self.specfemdir is not None:
                    lseis.createsimdir(self.specfemdir, _pardir,
                                       specfem_dict=self.specfem_dict)
                else:
                    self.__create_dir__(_pardir)

        # Write stations file
        lseis.inv2STATIONS(
            self.stations, os.path.join(self.synt_syntdir, "DATA", "STATIONS"))

        # Update Par_file depending on the parameter.
        syn_parfile = os.path.join(self.synt_syntdir, "DATA", "Par_file")
        syn_pars = lseis.read_parfile(syn_parfile)
        syn_pars["USE_SOURCE_DERIVATIVE"] = False

        # Adapt duration
        syn_pars["RECORD_LENGTH_IN_MINUTES"] = self.simulation_duration

        # Write Stuff to Par_file
        lseis.write_parfile(syn_pars, syn_parfile)

        # Do the same for the parameters to invert for.
        for _par, _pardir in self.synt_pardirs.items():

            # Half duration an time-shift don't need extra simulations
            if _par not in self.nosimpars:

                # Write stations file
                lseis.inv2STATIONS(
                    self.stations, os.path.join(_pardir, "DATA", "STATIONS"))

                # Update Par_file depending on the parameter.
                dsyn_parfile = os.path.join(_pardir, "DATA", "Par_file")
                dsyn_pars = lseis.read_parfile(dsyn_parfile)

                # Set data parameters and  write new parfiles
                locations = ["latitude", "longitude", "depth_in_m"]
                if _par in locations:
                    dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
                    if _par == "depth_in_m":
                        # 1 for depth
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1
                    elif _par == "latitude":
                        # 2 for latitude
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 2
                    else:
                        # 3 for longitude
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 3
                else:
                    dsyn_pars["USE_SOURCE_DERIVATIVE"] = False

                # Adapt duration
                dsyn_pars["RECORD_LENGTH_IN_MINUTES"] \
                    = self.simulation_duration

                # Write Stuff to Par_file
                lseis.write_parfile(dsyn_pars, dsyn_parfile)

    def __update_cmt__(self, model):
        """Given a scaled model vector update the current cmtsource.

        Parameters
        ----------
        model : ndarray
            scaled model vector
        """
        cmt = deepcopy(self.cmtsource)
        for _par, _modelval in zip(self.pars, model * self.scale):
            setattr(cmt, _par, _modelval)
        self.cmt_out = cmt

    def __write_sources__(self):
        """Writes the source files into the corresponding simulation
        directories for the synthetics and the frechet derivatives."""

        # Update cmt solution with new model values
        cmt = deepcopy(self.cmtsource)
        for _par, _modelval in zip(self.pars, self.model):
            setattr(cmt, _par, _modelval)

        # Writing synthetic CMT solution
        lutils.log_action("Writing Synthetic CMTSOLUTION",
                          plogger=self.logger.info)
        cmt.write_CMTSOLUTION_file(os.path.join(
            self.synt_syntdir, "DATA", "CMTSOLUTION"))

        # For the perturbations it's slightly more complicated.
        for _par, _pardir in self.synt_pardirs.items():

            if _par not in ["time_shift", "half_duration"]:
                # Write source to the directory of simulation
                lutils.log_action(
                    f"Writing Frechet CMTSOLUTION for {_par}",
                    plogger=self.logger.info)

                if self.pardict[_par]["pert"] is not None:
                    # Perturb source at parameter
                    cmt_pert = deepcopy(cmt)

                    # If parameter a part of the tensor elements then set the
                    # rest of the parameters to 0.
                    tensorlist = ['m_rr', 'm_tt', 'm_pp',
                                  'm_rt', 'm_rp', 'm_tp']
                    if _par in tensorlist:
                        for _tensor_el in tensorlist:
                            if _tensor_el != _par:
                                setattr(cmt_pert, _tensor_el, 0.0)
                            else:
                                setattr(cmt_pert, _tensor_el,
                                        self.pardict[_par]["pert"])
                    else:
                        # Get the parameter to be perturbed
                        to_be_perturbed = getattr(cmt_pert, _par)

                        # Perturb the parameter
                        to_be_perturbed += self.pardict[_par]["pert"]

                        # Set the perturb
                        setattr(cmt_pert, _par, to_be_perturbed)

                    cmt_pert.write_CMTSOLUTION_file(os.path.join(
                        _pardir, "DATA", "CMTSOLUTION"))
                else:
                    cmt.write_CMTSOLUTION_file(os.path.join(
                        _pardir, "DATA", "CMTSOLUTION"))

    def __run_simulations__(self):

        lutils.log_action("Submitting all simulations",
                          plogger=self.logger.info)
        # Initialize necessary commands
        cmd_list = self.nsim * [f'{self.launch_method} ./bin/xspecfem3D']

        cwdlist = [self.synt_syntdir]
        cwdlist.extend(
            [_pardir for _par, _pardir in self.synt_pardirs.items()
             if _par not in self.nosimpars])
        asyncio.run(lshell.asyncio_commands(cmd_list, cwdlist=cwdlist))

    def __run_forward_only__(self):

        # Initialize necessary commands
        lutils.log_action(
            "Submitting forward simulation", plogger=self.logger.info)
        cmd_list = [f'{self.launch_method} ./bin/xspecfem3D']
        cwdlist = [self.synt_syntdir]
        asyncio.run(lshell.asyncio_commands(cmd_list, cwdlist=cwdlist))

    def __run_parameters_only__(self):

        # Initialize necessary commands
        lutils.log_action(
            "Submitting parameter simulations", plogger=self.logger.info)
        cmd_list = (self.nsim - 1) * [f'{self.launch_method} ./bin/xspecfem3D']

        cwdlist = []
        cwdlist.extend(
            [_pardir for _par, _pardir in self.synt_pardirs.items()
             if _par not in self.nosimpars])
        asyncio.run(lshell.asyncio_commands(cmd_list, cwdlist=cwdlist))

    def forward(self, model):
        # Update model
        if self.zero_trace:
            self.model = model[:-1] * self.scale
            self.scaled_model = model[:-1]
        else:
            self.model = model * self.scale
            self.scaled_model = model

        # Write sources for next iteration
        self.__write_sources__()

        # Run forward simulation
        self.__run_forward_only__()

        # Process synthetic only
        self.process_synt()

    def compute_cost_gradient(self, model):

        # Update model
        self.model = model * self.scale
        self.scaled_model = model

        # Write sources for next iteration
        self.__write_sources__()

        # Run the simulations
        with lutils.Timer(plogger=self.logger.info):
            self.__run_simulations__()

        # Get streams
        self.process_all_synt()

        # Window Data
        if self.not_windowed_yet:
            self.__window__()
            self.not_windowed_yet = False

        return (
            self.__compute_cost__(), self.__compute_gradient__() * self.scale
        )

    def compute_cost_gradient_hessian(self, model):

        # Update model
        if self.zero_trace:

            # Read in model
            self.model = model[:-1] * self.scale

            # Fix depth to 5 km
            if self.model[self.model_idxdict['depth_in_m']] < self.mindepth:
                self.model[self.model_idxdict['depth_in_m']] = self.mindepth

            # Scale model again
            self.scaled_model = self.model / self.scale
        else:

            # Read in model
            self.model = model * self.scale

            # Fix depth to 5 km
            if self.model[self.model_idxdict['depth_in_m']] < self.mindepth:
                self.model[self.model_idxdict['depth_in_m']] = self.mindepth

            # Scale model again
            self.scaled_model = self.model / self.scale

        # Write sources for next iteration
        self.__write_sources__()

        # Run the simulations
        if self.iteration == 0:

            # First set of forward simulations where run when for windowing
            pass

        else:
            # Run simulations with the updated sources
            with lutils.Timer(plogger=self.logger.info):
                self.__run_simulations__()

            # Get streams
            self.process_all_synt()

        # Window Data
        if self.not_windowed_yet:
            self.__window__()
            self.not_windowed_yet = False

        # Evaluate
        cost = self.__compute_cost__()
        g, h = self.__compute_gradient_and_hessian__()

        # Raw hessian
        self.hessians.append(h.flatten())

        # Get normalization factor at the first iteration
        if self.iteration == 0:
            self.cost_norm = cost
            self.iteration = 1

        # Normalize the cost using the first cost calculation
        cost /= self.cost_norm

        self.logger.debug("Raw")
        self.logger.debug(f"C: {cost}")
        self.logger.debug("G:")
        self.logger.debug(g.flatten())
        self.logger.debug("H")
        self.logger.debug(h.flatten())

        # Scaling of the cost function
        g *= self.scale
        h = np.diag(self.scale) @ h @ np.diag(self.scale)

        # scaled hessian
        self.hessians_scaled.append(h.flatten())
        print(self.hessians_scaled)
        self.logger.debug("Scaled")
        self.logger.debug(f"C: {cost}")
        self.logger.debug("G:")
        self.logger.debug(g.flatten())
        self.logger.debug("H")
        self.logger.debug(h.flatten())

        if self.damping > 0.0:
            factor = self.damping * np.trace(h) / h.shape[0]
            modelres = self.scaled_model - self.init_scaled_model

            self.logger.debug(f"f: {factor}")
            self.logger.debug(f"Model Residual: {modelres.flatten()}")
            self.logger.debug(f"Cost: {cost}")

            g += factor * modelres
            h += factor * np.eye(len(self.model))

            self.logger.debug("Damped")
            self.logger.debug(f"C: {cost}")
            self.logger.debug("G:")
            self.logger.debug(g.flatten())
            self.logger.debug("H")
            self.logger.debug(h.flatten())

        elif self.hypo_damping > 0.0:

            # Only get the hessian elements of the hypocenter
            # hdiag = np.diag(h)[self.hypo_damp_index_array]
            # factor = self.hypo_damping * np.max(np.abs((hdiag)))
            factor = self.hypo_damping * np.trace(h) / h.shape[0]
            modelres = self.scaled_model - self.init_scaled_model

            self.logger.debug("HypoDiag:")
            self.logger.debug(np.diag(h))
            self.logger.debug("HypoDamping:")
            self.logger.debug(self.hypo_damping)
            self.logger.debug(f"f: {factor}")
            self.logger.debug(f"Model Residual: {modelres.flatten()}")

            # Add damping
            g += factor * modelres
            h += factor * np.diag(self.hypo_damp_array)

            self.logger.debug("Hypocenter-Damped")
            self.logger.debug(f"C: {cost}")
            self.logger.debug("G:")
            self.logger.debug(g.flatten())
            self.logger.debug("H")
            self.logger.debug(h.flatten())

        # Add zero trace condition
        if self.zero_trace:  # and not self.zero_energy:
            m, n = h.shape
            hz = np.zeros((m+1, n+1))
            hz[:-1, :-1] = h
            hz[:, -1] = self.zero_trace_array
            hz[-1, :] = self.zero_trace_array
            h = hz
            g = np.append(g, 0.0)
            g[-1] = np.sum(self.scaled_model[self.zero_trace_index_array])

            # Show stuf when debugging
            self.logger.debug("Constrained:")
            self.logger.debug(f"C: {cost}")
            self.logger.debug("G:")
            self.logger.debug(g.flatten())
            self.logger.debug("H")
            self.logger.debug(h.flatten())

        return cost, g, h

    def __compute_cost__(self):

        cost = 0
        for _wtype in self.processdict.keys():

            cgh = lseis.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                verbose=True if self.loglevel >= 20 else False,
                normalize=self.normalize,
                weight=self.weighting)
            cost += cgh.cost() * self.processdict[_wtype]["weight"]
        return cost

    def __compute_residuals__(self):

        residuals = dict()
        for _wtype in self.processdict.keys():

            cgh = lseis.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                verbose=True if self.loglevel >= 20 else False,
                normalize=self.normalize,
                weight=False)
            residuals[_wtype] = cgh.residuals()

        with open(os.path.join(self.cmtdir, "residuals.pkl"), "wb") as f:
            cPickle.dump(deepcopy(residuals), f)

        return residuals

    def __compute_gradient__(self):

        gradient = np.zeros_like(self.model)

        for _wtype in self.processdict.keys():
            # Get all perturbations
            dsyn = list()
            for _i, _par in enumerate(self.pardict.keys()):
                dsyn.append(self.synt_dict[_wtype][_par])

            # Create costgradhess class to computte gradient
            cgh = lseis.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                dsyn=dsyn,
                verbose=True if self.loglevel >= 20 else False,
                normalize=self.normalize,
                weight=self.weighting)

            gradient += cgh.grad() * self.processdict[_wtype]["weight"]

        return gradient

    def __compute_gradient_and_hessian__(self):

        gradient = np.zeros_like(self.model)
        hessian = np.zeros((len(self.model), len(self.model)))

        for _wtype in self.processdict.keys():

            # Get all perturbations
            dsyn = list()
            for _i, _par in enumerate(self.pardict.keys()):
                dsyn.append(self.synt_dict[_wtype][_par])

            # Create costgradhess class to computte gradient
            cgh = lseis.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                dsyn=dsyn,
                verbose=True if self.loglevel >= 20 else False,
                normalize=self.normalize,
                weight=self.weighting)

            tmp_g, tmp_h = cgh.grad_and_hess()
            gradient += tmp_g * self.processdict[_wtype]["weight"]
            hessian += tmp_h * self.processdict[_wtype]["weight"]

        self.logger.debug("M, G, H:")
        self.logger.debug(self.model)
        self.logger.debug(gradient.flatten())
        self.logger.debug(hessian.flatten())

        return gradient, hessian

    def plot_data(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            with PdfPages(os.path.join(
                    outputdir, f"data_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    fig = plot_seismograms(obsd_tr, cmtsource=self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages
                    # object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-Data-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def save_seismograms(self):

        outdir = os.path.join(self.cmtdir, "output")
        obsddir = os.path.join(outdir, "observed")
        syntdir = os.path.join(outdir, "synthetic")
        syntdir_init = os.path.join(outdir, "synthetic_init")
        stations = os.path.join(outdir, "STATIONS.xml")

        # Make directories
        if os.path.exists(outdir) is False:
            os.makedirs(outdir)
        if os.path.exists(obsddir) is False:
            os.makedirs(obsddir)
        if os.path.exists(syntdir) is False:
            os.makedirs(syntdir)

        # Write out stations
        self.stations.write(stations, format="STATIONXML")

        # Write processed data
        for _wtype, _stream in self.data_dict.items():

            filename = os.path.join(obsddir, f"{_wtype}_stream.pkl")
            with open(filename, 'wb') as f:
                cPickle.dump(_stream, f)

        # Write processed synthetics
        # Note that you have to run an extra siumulation the right model
        # to get the accruate
        for _wtype in self.synt_dict.keys():

            filename = os.path.join(syntdir, f"{_wtype}_stream.pkl")
            with open(filename, 'wb') as f:
                cPickle.dump(self.synt_dict[_wtype]["synt"], f)

        # Write processed initial synthetics
        if hasattr(self, "synt_dict_init"):
            if os.path.exists(syntdir_init) is False:
                os.makedirs(syntdir_init)
            for _wtype in self.synt_dict_init.keys():
                filename = os.path.join(syntdir_init, f"{_wtype}_stream.pkl")
                with open(filename, 'wb') as f:
                    cPickle.dump(self.synt_dict_init[_wtype]["synt"], f)

    def write_measurements(
            self, data: dict, synt: dict, post_fix: str = None):

        window_dict = get_all_measurements(
            data, synt, self.cmtsource, logger=self.logger)

        # Create output file
        filename = "measurements"
        if post_fix is not None:
            filename += "_" + post_fix
        filename += ".pkl"

        outfile = os.path.join(self.cmtdir, filename)
        with open(outfile, "wb") as f:
            cPickle.dump(window_dict, f)

        return window_dict

    def plot_station(self, network: str, station: str, outputdir="."):
        plt.switch_backend("pdf")
        # Get station data
        for _wtype in self.processdict.keys():
            try:
                obsd = self.data_dict[_wtype].select(
                    network=network, station=station)
                synt = self.synt_dict[_wtype]["synt"].select(
                    network=network, station=station)
            except Exception as e:
                self.logger.warning(
                    f"Could load station {network}{station} -- {e}")
            # Plot PDF for each wtype
            with PdfPages(os.path.join(
                    outputdir, f"{network}.{station}_{_wtype}.pdf")) as pdf:
                for component in ["Z", "R", "T"]:
                    try:
                        obsd_tr = obsd.select(
                            station=station, network=network,
                            component=component)[0]
                        synt_tr = synt.select(
                            station=station, network=network,
                            component=component)[0]
                    except Exception as err:
                        self.logger.warning(
                            f"Couldn't find obs or syn for NET.STA.COMP:"
                            f" {network}.{station}.{component} -- {err}")
                        continue

                    fig = plot_seismograms(obsd_tr, synt_tr, self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages
                    # object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_station_der(self, network: str, station: str, outputdir="."):
        plt.switch_backend("pdf")
        # Get station data
        for _wtype in self.processdict.keys():
            # Plot PDF for each wtype
            with PdfPages(os.path.join(
                    outputdir,
                    f"{network}.{station}_{_wtype}_derivatives.pdf")) as pdf:
                for _par in self.synt_dict[_wtype].keys():
                    if _par != "synt":
                        try:
                            synt = self.synt_dict[_wtype][_par].select(
                                network=network, station=station)
                        except Exception as e:
                            self.logger.warning(f"Could load station "
                                                f"{network}{station} -- {e}")
                        for component in ["Z", "R", "T"]:
                            try:
                                synt_tr = synt.select(
                                    station=station, network=network,
                                    component=component)[0]
                            except Exception as err:
                                self.logger.warning(
                                    f"Couldn't find obs or syn "
                                    f"for NET.STA.COMP:"
                                    f" {network}.{station}.{component} "
                                    f"-- {err}")
                                continue

                            fig = plot_seismograms(
                                synt_tr, cmtsource=self.cmtsource,
                                tag=f"{_wtype.capitalize()}-"
                                f"{_par.capitalize()}")
                            # saves the current figure into a pdf page
                            pdf.savefig()
                            plt.close(fig)

                    # We can also set the file's metadata via the PdfPages
                    # object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_windows(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            self.logger.info(f"Plotting {_wtype} waves")
            with PdfPages(os.path.join(
                    outputdir, f"windows_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    try:
                        synt_tr = self.synt_dict[_wtype]["synt"].select(
                            station=obsd_tr.stats.station,
                            network=obsd_tr.stats.network,
                            component=obsd_tr.stats.component)[0]
                    except Exception as err:
                        self.logger.warning(err)
                        self.logger.warning(
                            "Couldn't find corresponding synt for "
                            f"obsd trace({obsd_tr.id}): {err}")
                        continue

                    fig = plot_seismograms(
                        obsd_tr, synt_tr, cmtsource=self.cmtsource, tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages
                    #  object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_final_windows(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            self.logger.info(f"Plotting {_wtype} waves")
            with PdfPages(os.path.join(
                    outputdir, f"final_windows_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    try:
                        synt_tr = self.synt_dict[_wtype]["synt"].select(
                            station=obsd_tr.stats.station,
                            network=obsd_tr.stats.network,
                            component=obsd_tr.stats.component)[0]
                        init_synt_tr \
                            = self.synt_dict_init[_wtype]["synt"].select(
                                station=obsd_tr.stats.station,
                                network=obsd_tr.stats.network,
                                component=obsd_tr.stats.component)[0]
                    except Exception as err:
                        self.logger.warning(
                            "Couldn't find corresponding synt for "
                            f"obsd trace({obsd_tr.id}): {err}")
                        continue

                    fig = plot_seismograms(
                        obsd_tr, init_synt_tr, synt_tr, self.cmtsource,
                        tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages
                    # object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    @ staticmethod
    def __create_dir__(dir, overwrite=False):
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        else:
            if overwrite:
                shutil.rmtree(dir)
                os.mkdir(dir)
            else:
                pass


def plot_seismograms(obsd: Trace, synt: Union[Trace, None] = None,
                     syntf: Union[Trace, None] = None,
                     cmtsource: Union[lseis.CMTSource, None] = None,
                     tag: Union[str, None] = None):
    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location

    trace_id = f"{network}.{station}.{location}.{channel}"

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
        if isinstance(synt, Trace):
            offset_synt = synt.stats.starttime - cmtsource.cmt_time
        if isinstance(syntf, Trace):
            offset_syntf = syntf.stats.starttime - cmtsource.cmt_time

    times = [offset + obsd.stats.delta * i for i in range(obsd.stats.npts)]
    if isinstance(synt, Trace):
        times_synt = [offset_synt + synt.stats.delta * i
                      for i in range(synt.stats.npts)]
    if isinstance(syntf, Trace):
        times_syntf = [offset_syntf + syntf.stats.delta * i
                       for i in range(syntf.stats.npts)]

    # Figure Setup
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(211)
    plt.subplots_adjust(left=0.075, right=0.925, top=0.95)

    ax1.plot(times, obsd.data, color="black", linewidth=0.75,
             label="Obs")
    if isinstance(synt, Trace):
        ax1.plot(times_synt, synt.data, color="red", linewidth=0.75,
                 label="Syn")
    if isinstance(syntf, Trace):
        ax1.plot(times_syntf, syntf.data, color="blue", linewidth=0.75,
                 label="New Syn")
    scaleabsmax = 1.25*np.max(np.abs(obsd.data))
    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(-scaleabsmax, scaleabsmax)
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    ax1.tick_params(labelbottom=False, labeltop=False)

    # Setting top left corner text manually
    if isinstance(tag, str):
        label = f"{trace_id}\n{tag.capitalize()}"
    else:
        label = f"{trace_id}"
    lplt.plot_label(ax1, label, location=1, dist=0.005, box=False)

    # plot envelope
    ax2 = plt.subplot(212)
    obsenv = lmat.envelope(obsd.data)
    ax2.plot(times, obsenv, color="black",
             linewidth=1.0, label="Obs")
    if isinstance(synt, Trace):
        ax2.plot(times, lmat.envelope(synt.data), color="red", linewidth=1,
                 label="Syn")
    if isinstance(syntf, Trace):
        ax2.plot(times, lmat.envelope(syntf.data), color="blue", linewidth=1,
                 label="New Syn")
    envscaleabsmax = 1.25*np.max(np.abs(obsenv))
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(0, envscaleabsmax)
    ax2.set_xlabel("Time [s]", fontsize=13)
    lplt.plot_label(ax2, "Envelope", location=1, dist=0.005, box=False)
    if isinstance(synt, Trace):
        try:
            for win in obsd.stats.windows:
                left = times[win.left]
                right = times[win.right]
                re1 = Rectangle((left, ax1.get_ylim()[0]), right - left,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax1.add_patch(re1)
                re2 = Rectangle((left, ax2.get_ylim()[0]), right - left,
                                ax2.get_ylim()[1] - ax2.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax2.add_patch(re2)
        except Exception as e:
            print(e)

    return fig


def bin():

    import sys
    import argparse

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='event', help='CMTSOLUTION file',
                        type=str)
    parser.add_argument('-i', '--inputfile', dest='inputfile',
                        help='Input file location',
                        required=False, type=str, default=None)
    parser.add_argument('-d', '--download_only', dest='download_only',
                        help='Download only', action='store_true',
                        required=False, default=False)
    args = parser.parse_args()

    cmtsolutionfile = args.event
    inputfile = args.inputfile

    # Get Input parameters
    if inputfile is None:
        inputdict = lutils.read_yaml_file(
            os.path.join(scriptdir, "input.yml"))
    else:
        inputdict = lutils.read_yaml_file(inputfile)

    # Get process params
    if inputdict["processparams"] is None:
        processdict = lutils.read_yaml_file(
            os.path.join(scriptdir, "process.yml"))
    else:
        processdict = lutils.read_yaml_file(inputdict["processparams"])

    # Set params
    pardict = inputdict["parameters"]
    database = inputdict["database"]
    specfem = inputdict["specfem"]
    launch_method = inputdict["launch_method"]
    download_data = inputdict["download_data"]
    hypo_damping = inputdict["hypo_damping"]
    damping = inputdict["damping"]
    duration = inputdict["duration"]
    overwrite = inputdict["overwrite"]
    zero_trace = inputdict["zero_trace"]
    start_label = inputdict["start_label"]
    solution_label = inputdict["solution_label"]

    if args.download_only:
        download_data = True
    else:
        # Set CPU Affinity
        lutils.reset_cpu_affinity(verbose=True)

    # Setup the download
    gcmt3d = GCMT3DInversion(
        cmtsolutionfile,
        databasedir=database,
        specfemdir=specfem,
        pardict=pardict,
        processdict=processdict,
        download_data=download_data,
        zero_trace=zero_trace,
        duration=duration,
        overwrite=overwrite,
        launch_method=launch_method,
        damping=damping,
        hypo_damping=hypo_damping,
        start_label=start_label,
        multiprocesses=40)

    if args.download_only:
        sys.exit()

    # gcmt3d.init()
    gcmt3d.process_data()
    gcmt3d.get_windows()
    gcmt3d.__compute_weights__()

    optim_list = []

    with lutils.Timer(plogger=gcmt3d.logger.info):

        # Gauss Newton Optimization Structure
        lutils.log_bar("GN", plogger=gcmt3d.logger.info)
        optim_gn = linv.Optimization("gn")
        optim_gn.logger = gcmt3d.logger.info
        optim_gn.compute_cost_and_grad_and_hess \
            = gcmt3d.compute_cost_gradient_hessian

        # Set attributes depending on the optimization input parameters
        for key, val in inputdict["optimization"].items():
            setattr(optim_gn, key, val)

        # Run optimization
        with lutils.Timer(plogger=gcmt3d.logger.info):
            optim_out = gcmt3d.optimize(optim_gn)
            lutils.log_action("DONE with Gauss-Newton.",
                              plogger=gcmt3d.logger.info)

        gcmt3d.logger.info("Shape:")
        gcmt3d.logger.info(optim_out.model.shape)
        gcmt3d.logger.info("Size:")
        gcmt3d.logger.info(optim_out.model.size)
        # Update model and write model
        if gcmt3d.zero_trace:
            gcmt3d.__update_cmt__(optim_out.model[:-1])
        else:
            gcmt3d.__update_cmt__(optim_out.model)

        # Write model to file
        gcmt3d.cmt_out.write_CMTSOLUTION_file(
            f"{gcmt3d.cmtdir}/{gcmt3d.cmt_out.eventname}_{solution_label}")

        optim_list.append(deepcopy(optim_out))

    print("Hessian list", gcmt3d.hessians_scaled)

    # Stuff for L-Curves
    # Get model related things to save
    if gcmt3d.zero_trace:
        init_model = optim_out.model_ini[:-1]
        model = optim_out.model[:-1]
        modelnorm = np.sqrt(np.sum(optim_out.model[:-1]**2))
        dmnorm = np.sqrt(np.sum((optim_out.model[:-1])**2))
        modelhistory = optim_out.msave[:-1, :]
        hessianhistory = optim_out.hsave
        scale = gcmt3d.scale[:-1]

        # Fix its shape
        hessianhistory = hessianhistory.reshape(
            (optim_out.n, optim_out.n, optim_out.nb_mem))[:-1, :-1, :]

    else:
        init_model = optim_out.model_ini
        model = optim_out.model
        modelnorm = np.sqrt(np.sum(optim_out.model**2))
        dmnorm = np.sqrt(np.sum(optim_out.model**2))
        modelhistory = optim_out.msave
        hessianhistory = optim_out.hsave
        scale = gcmt3d.scale

    hdidx = gcmt3d.hypo_damp_index_array
    cost = optim_out.fcost
    fcost_hist = optim_out.fcost_hist
    fcost_init = optim_out.fcost_init

    # Save to npz file
    np.savez(
        os.path.join(gcmt3d.cmtdir, "summary.npz"),
        cost=cost,
        init_model=init_model,
        modelnorm=modelnorm,
        model=model,
        dmnorm=dmnorm,
        modelhistory=modelhistory,
        fcost_hist=fcost_hist,
        fcost_init=fcost_init,
        scale=scale,
        damping=damping,
        hypo_damping=hypo_damping,
        hypo_damping_index=hdidx,
        hessianhistory=hessianhistory,
        hessians=np.array(gcmt3d.hessians),
        hessians_scaled=np.array(gcmt3d.hessians_scaled)
    )

    # To be able to output the current model we need to go back and run one
    # iteration with tht current model
    gcmt3d.forward(optim_out.model)

    # Then compute and save the measurements
    gcmt3d.write_measurements(
        gcmt3d.data_dict, gcmt3d.synt_dict_init, post_fix="before")
    gcmt3d.write_measurements(
        gcmt3d.data_dict, gcmt3d.synt_dict, post_fix="after")

    try:
        gcmt3d.save_seismograms()
    except Exception as e:
        print(e)

    try:
        gcmt3d.plot_final_windows(outputdir=gcmt3d.cmtdir)
    except Exception as e:
        print(e)

    # # Write PDF
    plt.switch_backend("pdf")
    # linv.plot_model_history(
    #     optim_list,
    #     list(pardict.keys()),  # "BFGS-R" "BFGS",
    #     outfile=f"{gcmt3d.cmtdir}/InversionHistory.pdf")
    linv.plot_optimization(
        optim_list,
        outfile=f"{gcmt3d.cmtdir}/misfit_reduction_history.pdf")


def bin_process_final():

    import argparse
    import sys

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='eventdir', help='event directory',
                        type=str)
    parser.add_argument('-i', '--inputfile', dest='inputfile',
                        help='Input file location',
                        required=False, type=str, default=None)
    parser.add_argument('-l', '--label', dest='label',
                        help='label for cmt in dir',
                        required=False, type=str, default=None)
    args = parser.parse_args()

    # Check if soluttion exitst
    cmtdir = args.eventdir
    cmtid = os.path.basename(cmtdir)
    label = args.label

    if label is not None:
        cmtsolutionfile = os.path.join(cmtdir, cmtid + f"_{label}")
    else:
        cmtsolutionfile = os.path.join(cmtdir, cmtid + "_gcmt")

    if os.path.exists(cmtsolutionfile) is False:
        print(f"CMT file {cmtsolutionfile} not found.")
        sys.exit()

    inputfile = args.inputfile

    # Get Input parameters
    if inputfile is None:
        inputdict = lutils.read_yaml_file(
            os.path.join(scriptdir, "input.yml"))
    else:
        inputdict = lutils.read_yaml_file(inputfile)

    # Get process params
    if inputdict["processparams"] is None:
        processdict = lutils.read_yaml_file(
            os.path.join(scriptdir, "process.yml"))
    else:
        processdict = lutils.read_yaml_file(inputdict["processparams"])

    # Set params
    pardict = inputdict["parameters"]
    database = inputdict["database"]
    specfem = inputdict["specfem"]
    launch_method = inputdict["launch_method"]
    download_data = inputdict["download_data"]
    hypo_damping = inputdict["hypo_damping"]
    damping = inputdict["damping"]
    duration = inputdict["duration"]
    overwrite = inputdict["overwrite"]
    zero_trace = inputdict["zero_trace"]
    start_label = label

    # Set CPU Affinity
    lutils.reset_cpu_affinity(verbose=True)

    # Setup the download
    gcmt3d = GCMT3DInversion(
        cmtsolutionfile,
        databasedir=database,
        specfemdir=specfem,
        pardict=pardict,
        processdict=processdict,
        download_data=download_data,
        zero_trace=zero_trace,
        duration=duration,
        overwrite=overwrite,
        launch_method=launch_method,
        damping=damping,
        hypo_damping=hypo_damping,
        start_label=start_label,
        multiprocesses=40)

    # Load and process the synthetic and
    gcmt3d.process_data()
    gcmt3d.process_synt()

    # Copy the initial synthetics
    gcmt3d.copy_init_synt()

    # Window the data
    with lutils.Timer(plogger=gcmt3d.logger.info):
        gcmt3d.__window__()
        gcmt3d.__compute_weights__()

    try:
        gcmt3d.save_seismograms()

    except Exception as e:
        print(e)
