import os
from glob import glob
from typing import Optional
from future.utils import lmap
from obspy import UTCDateTime, Stream
import logging
import numpy as np
from lwsspy import plot as lplot
from lwsspy import signal as lsig
from lwsspy import seismo as lseis
from lwsspy import maps as lmap
from .io import read_measurements_label


def get_toffset(
        tsample: int, dt: float, t0: UTCDateTime, origin: UTCDateTime) -> float:
    """Computes the time of a sample with respect to origin time

    Parameters
    ----------
    tsample : int
        sample on trace
    dt : float
        sample spacing
    t0 : UTCDateTime
        time of the first sample
    origin : UTCDateTime
        origin time

    Returns
    -------
    float
        Time relative to origin time
    """

    # Second on trace
    trsec = (tsample*dt)
    return (t0 + trsec) - origin


def get_measurements_and_windows(
        obs: Stream, syn: Stream, event: lseis.CMTSource, logger: logging.Logger):
    """Make measurements on two correpsonding streams.

    Parameters
    ----------
    obs : Stream
        Observed stream
    syn : Stream
        synthetic stream
    event : lseis.CMTSource
        event
    logger : logging.Logger, optional
        Logger. Default None


    Returns
    -------
    dict
        dictionary with measurements for each component.
    """

    if logger is None:
        logger = logging.getLogger('lwsspy')

    windows = dict()

    # Create dict to access traces
    for _component in ["R", "T", "Z"]:
        windows[_component] = dict()
        windows[_component]["id"] = []
        windows[_component]["dt"] = []
        windows[_component]["starttime"] = []
        windows[_component]["endtime"] = []
        windows[_component]["nsamples"] = []
        windows[_component]["latitude"] = []
        windows[_component]["longitude"] = []
        windows[_component]["distance"] = []
        windows[_component]["azimuth"] = []
        windows[_component]["back_azimuth"] = []
        windows[_component]["nshift"] = []
        windows[_component]["time_shift"] = []
        windows[_component]["maxcc"] = []
        windows[_component]["dlna"] = []
        windows[_component]["L1"] = []
        windows[_component]["L2"] = []
        windows[_component]["dL1"] = []
        windows[_component]["dL2"] = []
        windows[_component]["trace_energy"] = []
        windows[_component]["L1_Power"] = []
        windows[_component]["L2_Power"] = []
        windows[_component]["corr_ratio"] = []

        for _tr in obs:
            if _tr.stats.component == _component \
                    and "windows" in _tr.stats:

                d = _tr.data
                try:
                    network, station, component = (
                        _tr.stats.network, _tr.stats.station,
                        _tr.stats.component)
                    s = syn.select(
                        network=network, station=station,
                        component=component)[0].data
                except Exception as e:
                    logger.warning(
                        f"{network}.{station}..{component}")
                    logger.error(e)
                    continue

                trace_energy = 0
                for win in _tr.stats.windows:
                    # Get window data
                    wd = d[win.left:win.right]
                    ws = s[win.left:win.right]

                    # Infos
                    dt = _tr.stats.delta
                    npts = _tr.stats.npts
                    winleft = get_toffset(
                        win.left, dt, win.time_of_first_sample,
                        event.origin_time)
                    winright = get_toffset(
                        win.right, dt, win.time_of_first_sample,
                        event.origin_time)

                    # Measurements
                    max_cc_value, nshift = lsig.xcorr(wd, ws)

                    # Get fixed window indeces.
                    try:
                        istart, iend = win.left, win.right
                        istart_d, iend_d, istart_s, iend_s = \
                            lsig.correct_window_index(
                                istart, iend, nshift, npts)
                        wd_fix = d[istart_d:iend_d]
                        ws_fix = s[istart_s:iend_s]
                    except ValueError as ve:
                        logger.warning(
                            f"Window [{winleft}, {winright}] on trace {_tr.id} "
                            f"was not taken into account: {ve}"
                        )
                        continue

                    # Populate the dictionary
                    windows[_component]["id"].append(_tr.id)
                    windows[_component]["dt"].append(dt)
                    windows[_component]["starttime"].append(winleft)
                    windows[_component]["endtime"].append(winright)
                    windows[_component]["latitude"].append(
                        _tr.stats.latitude
                    )
                    windows[_component]["longitude"].append(
                        _tr.stats.longitude
                    )
                    windows[_component]["distance"].append(
                        _tr.stats.distance
                    )
                    windows[_component]["azimuth"].append(
                        _tr.stats.azimuth
                    )
                    windows[_component]["back_azimuth"].append(
                        _tr.stats.back_azimuth
                    )

                    powerl1 = lsig.power_l1(wd, ws)
                    powerl2 = lsig.power_l2(wd, ws)
                    norm1 = lsig.norm1(wd)
                    norm2 = lsig.norm2(wd)
                    dnorm1 = lsig.dnorm1(wd, ws)
                    dnorm2 = lsig.dnorm2(wd, ws)
                    dlna = lsig.dlna(wd_fix, ws_fix)
                    trace_energy += norm2

                    windows[_component]["L1"].append(norm1)
                    windows[_component]["L2"].append(norm2)
                    windows[_component]["dL1"].append(dnorm1)
                    windows[_component]["dL2"].append(dnorm2)
                    windows[_component]["dlna"].append(dlna)
                    windows[_component]["L1_Power"].append(powerl1)
                    windows[_component]["L2_Power"].append(powerl2)
                    windows[_component]["nshift"].append(nshift)
                    windows[_component]["time_shift"].append(
                        nshift * dt
                    )
                    windows[_component]["maxcc"].append(
                        max_cc_value
                    )
                    windows[_component]["corr_ratio"].append(
                        np.sum(wd_fix * ws_fix)/np.sum(ws_fix ** 2)
                    )
                # Create array with the energy
                windows[_component]["trace_energy"].extend(
                    [trace_energy]*len(_tr.stats.windows))

    return windows


def get_all_measurements(
        datadict: dict, syntdict: dict, event: lseis.CMTSource,
        logger: Optional[logging.Logger] = None):

    window_dict = dict()

    for _wtype, _obs_stream in datadict.items():

        # Get corresponding Synthetic data
        _syn_stream = syntdict[_wtype]["synt"]

        window_dict[_wtype] = get_measurements_and_windows(
            _obs_stream, _syn_stream, event, logger=logger)

    return window_dict


def get_measurement_N(
        database0: str, label0: str,
        database1: str, label1: str,
        v: bool = True):
    """Takes in databse locations and labels to create a table that contains
    measurement count vs. parameter change.

    Parameters
    ----------
    database0 : str
        Starting database
    label0 : str
        label of starting solution
    database1 : str
        Final database
    label1 : str
        label of final solution
    v : bool, optional
        flag to turn on verbose output
    Returns
    -------
    Arraylike table
        CID, ddepth, dM0, dcmt, dx, *[measurement counts for wave types]

    """

    # Get all cmtfiles
    cmtfiles1 = glob(os.path.join(database1, f"*/*_{label1}"))
    N = len(cmtfiles1)

    # Create catalogs
    cat1 = lseis.CMTCatalog(cmtfiles1)

    # Waves and components
    mtype = 'dlna'  # placeholder measurement
    waves = ['body', 'surface', 'mantle']
    comps = ['Z', 'R', 'T']
    wcomb = [f"{w}-{c}" for w in waves for c in comps]

    # Create numpy structure dtype
    dtypes = np.dtype([
        ('event', str),
        ('dz', np.float64),
        ('dM0', np.float64),
        ('dt', np.float64),
        ('dx', np.float64),
        *[(wc, np.float64) for wc in wcomb]
    ])

    Nm = []
    for cmtfile1 in cmtfiles1:

        if v:
            print(f"Adding {cmtfile1} ...")

        # Load Final solution
        cmt1 = lseis.CMTSource.from_CMTSOLUTION_file(cmtfile1)

        # Create Filename for the start solution and load it
        cmtfile0 = os.path.join(
            database0, cmt1.eventname, f"{cmt1.eventname}_{label0}")
        cmt0 = lseis.CMTSource.from_CMTSOLUTION_file(cmtfile0)

        # Compute changes
        dz = (cmt1.depth_in_m - cmt0.depth_in_m)/1000.0
        dt = cmt1.time_shift - cmt0.time_shift
        dM0 = (cmt1.M0 - cmt0.M0)/cmt0.M0
        dx = lmap.haversine(
            cmt0.longitude, cmt0.latitude, cmt1.longitude, cmt1.latitude)

        # Get number of measurements involved
        d = read_measurements_label(os.path.dirname(cmtfile1), label1)

        # Empty list to be used for the table
        mlist = 9 * [np.nan]

        # Fill if possible
        if isinstance(d, dict):

            counter = 0
            for w in ['body', 'surface', 'mantle']:
                if w not in d:
                    counter += 3
                    continue
                for c in ['R', 'T', 'Z']:
                    mlist[counter] = len(d[w][c][mtype])
                    counter += 1

        Nm.append(
            (cmt1.eventname, dz, dM0, dt, dx, *mlist)
        )

    return np.array(Nm, dtype=dtypes)
