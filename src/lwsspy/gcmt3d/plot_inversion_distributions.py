
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.seismo import CMTCatalog
from lwsspy.plot import pick_colors_from_cmap
import matplotlib
from copy import deepcopy


def tensor2M0(tensor):
    """
    Scalar Moment M0 in Nm
    """
    return (tensor[:, 0] ** 2 + tensor[:, 1] ** 2 + tensor[:, 2] ** 2
            + 2 * tensor[:, 3] ** 2 + 2 * tensor[:, 4] ** 2
            + 2 * tensor[:, 5] ** 2) ** 0.5 * 0.5 ** 0.5


def get_stuff(inversionfile, scaling_file, catalog) -> tuple:

    # Get inversion file
    v = np.load(inversionfile)

    # Grab variables from the inversion File
    events = v['events']
    G = v['G']
    H = v['H']

    # Grab scaling
    scaling = np.loadtxt(scaling_file)

    # Load catalog
    cat = CMTCatalog.load(catalog)
    cat = cat.in_catalog(events)

    return events, G, H, scaling, cat


def inversion_tests(events, G, H, scaling, cat: CMTCatalog, damp_type='hypo',
                    zerotrace: bool = True):
    """[summary]

    Parameters
    ----------
    events : list of events
        list of events
    G : ndarray
        gradients
    H : ndarray
        Hessians
    scaling : ndarray
        scaling of model parameters
    cat : CMTCatalog
        catalog containing original events
    damp_type : str, optional
        damping types, 'all' damps all events equally, 'depth' damps full
        hessian for all events with depth <70km depth, 'hypo' damps only the
        hypocenter parameters, by default 'hypo'.
    zerotrace : bool
        Whether the inversion should be constrained to keep the trace of the
        moment tensor zerotrace.

    Returns
    -------
    tuple of
        (change in model parameters for each event and damping value,
         list of damping values)
    """

    # Damping list
    damping_list = np.logspace(-4, -2, 7)

    # Initialize an array for model changes for all events and all dampings
    dm = np.zeros((len(damping_list), *G.shape))

    # Scaling array
    scale = np.ones_like(G)

    # Depth
    depth = cat.getvals(vtype="depth_in_m")

    # M0
    M0 = cat.getvals(vtype="M0")

    # Compute Trace for entire Array
    tr = np.trace(H, axis1=1, axis2=2)

    # Damping diagonal matrix
    fulldampd = np.diag(np.ones(10))
    hypodampd = np.diag(np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0]))

    # Create scaling array
    s = scaling
    sdiag = np.diag(scaling)

    # Get Scalar moment of each event
    M0 = cat.getvals(vtype='M0')

    # Initialize model vectors
    dm = np.zeros((len(damping_list), *G.shape))

    # Zero trace params
    zero_trace = True
    pos = np.arange(10)
    m, n = H[0, :, :].shape

    # Zero trace things
    if zerotrace:
        fulldampd = np.diag(np.ones(11))
        fulldampd[-1] = 0
        hypodampd = np.diag(
            np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0]))
        h = np.zeros((m+1, n+1))
        g = np.zeros(m+1)
        zero_trace_array = np.zeros(11)
        zero_trace_array[:3] = 1
        h[:, -1] = zero_trace_array
        h[-1, :] = zero_trace_array
        zp = slice(0, -1)
    else:
        fulldampd = np.diag(np.ones(10))
        hypodampd = np.diag(np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0]))
        d = np.diag(np.ones(10))
        h = np.zeros((m, n))
        g = np.zeros(m)
        zp = slice(0, m)

    # Dampstyle array
    if damp_type == 'all':
        fulldamp = np.ones(G.shape[0])
        hypodamp = np.zeros(G.shape[0])
    elif damp_type == 'hypo':
        fulldamp = np.zeros(G.shape[0])
        hypodamp = np.ones(G.shape[0])
    elif damp_type == 'depth':
        fulldamp = np.zeros(G.shape[0])
        hypodamp = np.zeros(G.shape[0])
        fulldamp[np.where(depth <= 35000)[0]] = 1
        hypodamp[np.where(depth > 35000)[0]] = 1
        print("\n\n", len(np.where(fulldamp == 1)[0]),
              'v.', len(np.where(hypodamp == 1)[0]), end='\n\n')
    else:
        raise ValueError('damping type not implemented.')

    for _i, (_h, _g, m0, _cmt) in enumerate(zip(H, G, M0, cat)):

        # Fix scaling
        s[:6] = m0
        sdiag[pos[:6], pos[:6]] = m0

        # Computed scaled gradient and hessian
        g[zp] = _g * s
        h[zp, zp] = sdiag @ _h @ sdiag

        # Zero trace requires the sum of things
        if zero_trace:
            g[-1] = np.sum(np.array([_cmt.tensor[:3]]))/m0

        for _j, _damp in enumerate(damping_list):
            print(
                f"{100*(_i+1)/G.shape[0]:3.0f}% -- {events[_i]:16} -- {_damp:10f}", end='\r')
            dm[_j, _i, :] = np.linalg.solve(
                h
                + _damp * tr[_i] * fulldamp[_i] * fulldampd
                + _damp * tr[_i] * hypodamp[_i] * hypodampd,
                -g)[:10] * s

    return dm, damping_list


def inversion_tests_depth(events, G, H, scaling, cat: CMTCatalog, damp_type='hypo'):
    """[summary]

    Parameters
    ----------
    events : list of events
        list of events
    G : ndarray
        gradients
    H : ndarray
        Hessians
    scaling : ndarray
        scaling of model parameters
    cat : CMTCatalog
        catalog containing original events
    damp_type : str, optional
        damping types, 'all' damps all events equally, 'depth' damps full
        hessian for all events with depth <70km depth, 'hypo' damps only the
        hypocenter parameters, by default 'hypo'.

    Returns
    -------
    tuple of
        (change in model parameters for each event and damping value,
         list of damping values)
    """

    # Damping list
    damping_list = np.logspace(-4, 0,   7)

    # Initialize an array for model changes for all events and all dampings
    dm = np.zeros((len(damping_list), *G.shape))

    # Initalize M0 array
    M0 = np.zeros(G.shape[0])

    # Scaling array
    scale = np.ones_like(G)

    # Damping diagonal matrix
    fulldampd = np.diag(np.ones(2))

    for _i, _cmt in enumerate(cat):

        M0[_i] = _cmt.M0
        scale[_i, :] = scaling
        g = G[_i, :] * scale[_i, :]
        h = np.diag(scale[_i, :]) @ H[_i, :, :] @ np.diag(scale[_i, :])

        for _j, _damp in enumerate(damping_list):
            print(
                f"{100*(_i+1)/G.shape[0]:3.0f}% -- {events[_i]:16} -- {_damp:10f}", end='\r')
            dm[_j, _i, :] = np.linalg.solve(
                h
                + _damp * np.trace(h) * fulldampd,
                -g)

    dms = dm * scale

    return dms, damping_list


def plot_results(dms, damping_list, cat: CMTCatalog, titles=None):

    if isinstance(dms, list) is False:
        dms = [deepcopy(dms)]
    else:
        dms = deepcopy(dms)

    Nrows = len(dms)
    plt.figure(figsize=(8, 1 + Nrows*1.25))

    labels = ['$M_{rr}$', '$M_{tt}$', '$M_{pp}$', '$M_{rt}$',
              '$M_{rp}$', '$M_{tp}$', '$t_{cmt}$', '$z$', '$\\theta$', '$\phi$']
    xlabels = [
        '$d\ln M_{rr}$', '$d\ln M_{tt}$', '$d\ln M_{pp}$', '$d\ln M_{rt}$',
        '$d\ln M_{rp}$', '$d\ln M_{tp}$',
        '$\delta t_{cmt}$ [s]', '$\delta z$ [km]', '$\delta \\theta$ [deg]', '$\delta \phi$ [deg]']
    # Get normalizaton
    M0 = cat.getvals(vtype='M0')

    matplotlib.rcParams.update(
        {
            'xtick.labelsize': 'xx-small',
            'ytick.labelsize': 'xx-small',
        }
    )
    # Get color
    colors = pick_colors_from_cmap(len(damping_list), 'rainbow')

    for i in range(len(dms)):

        # Get qunatiles of wides range
        dms[i][:, :, :6] = dms[i][:, :, :6]/M0[None, :, None]
        dms[i][:, :, 7] = dms[i][:, :,  7]/1000.0

    xrange = (
        np.quantile(dms[0][0, :, :], 0.02, axis=0),
        np.quantile(dms[0][0, :, :], 0.98, axis=0)
    )

    label_format = '{:6.2f}'
    for i in range(len(dms)):

        for j in range(10):

            ax = plt.subplot(Nrows, 11, 11*i + (1+j))

            if i == 0:
                plt.title(labels[j])
            pc = []
            for k, _damp in enumerate(damping_list):

                if i == 0:
                    pc.append(
                        plt.plot([], [], label=f"{_damp}", c=colors[k])[0])

                plt.hist(
                    dms[i][k, :, j], bins=200, range=(xrange[0][j], xrange[1][j]),
                    histtype='stepfilled', edgecolor=colors[k],
                    facecolor='none', density=True)

            ax.vlines(
                0, 0, ax.get_ylim()[1], color='k', ls=':', lw=0.75, zorder=0
            )
            ax.spines.top.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)

            x0, x1 = ax.get_xlim()
            visible_ticks = [t for t in ax.get_xticks() if t >= x0 and t <= x1]
            ax.set_xticks(visible_ticks)

            if i == len(dms)-1:
                ax.set_xticklabels(
                    [label_format.format(x) for x in ax.get_xticks()],
                    rotation=45, ha="right")
                ax.set_xlabel(xlabels[j], fontsize='x-small')
            else:
                # ax.set_xticklabels([])
                ax.tick_params(labelbottom=False)
                pass
            if j == 0:
                plt.ylabel(titles[i].capitalize(),
                           rotation=0, ha='right', va='baseline')

            ax.tick_params(which='both', left=False, right=False,
                           labelleft=False, labelright=False, top=False,
                           labeltop=False)
            # ax.tick_params(which='minor', left=False, right=False,
            #                labelleft=False, labelright=False, top=False, labeltop=False)

        if i == 0:
            ax = plt.subplot(Nrows, 11, 11)
            ax.axis('off')
            ax.legend(pc, [f"{_damp:.4f}" for _damp in damping_list], loc='upper right',
                      fontsize='xx-small', title='Damping', frameon=False,
                      borderaxespad=0.0, title_fontsize='x-small')

    plt.subplots_adjust(hspace=0.075, wspace=0.125, left=0.075,
                        right=0.975, bottom=0.125, top=0.9)


def plot_results_depth(dms, damping_list, cat: CMTCatalog, titles=None):

    if isinstance(dms, list) is False:
        dms = [deepcopy(dms)]
    else:
        dms = deepcopy(dms)

    if not titles:
        titles = [""] * len(dms)

    Nrows = len(dms)
    plt.figure(figsize=(5, 2))

    labels = ['$t_{cmt}$', '$z$']
    xlabels = ['$\delta t_{cmt}$ [s]', '$\delta z$ [km]']

    matplotlib.rcParams.update(
        {
            'xtick.labelsize': 'xx-small',
            'ytick.labelsize': 'xx-small',
        }
    )

    # Get color
    colors = pick_colors_from_cmap(len(damping_list), 'rainbow')

    for i in range(len(dms)):

        # Get qunatiles of wides range
        dms[i][:, :, 1] = dms[i][:, :,  1]/1000.0

    xrange = (
        np.quantile(dms[0][0, :, :], 0.02, axis=0),
        np.quantile(dms[0][0, :, :], 0.98, axis=0)
    )

    label_format = '{:6.2f}'
    for i in range(len(dms)):

        for j in range(2):

            ax = plt.subplot(Nrows, 3, 3*i + (1+j))

            if i == 0:
                plt.title(labels[j])
            pc = []
            for k, _damp in enumerate(damping_list):

                if i == 0:
                    pc.append(
                        plt.plot([], [], label=f"{_damp}", c=colors[k])[0])

                plt.hist(
                    dms[i][k, :, j], bins=150, range=(xrange[0][j], xrange[1][j]),
                    histtype='stepfilled', edgecolor=colors[k],
                    facecolor='none', density=True)

            ax.vlines(
                0, 0, ax.get_ylim()[1], color='k', ls=':', lw=0.75, zorder=0
            )
            ax.spines.top.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)

            x0, x1 = ax.get_xlim()
            visible_ticks = [t for t in ax.get_xticks() if t >= x0 and t <= x1]
            ax.set_xticks(visible_ticks)

            if i == len(dms)-1:
                ax.set_xticklabels(
                    [label_format.format(x) for x in ax.get_xticks()],
                    rotation=45, ha="right")
                ax.set_xlabel(xlabels[j], fontsize='x-small')
            else:
                # ax.set_xticklabels([])
                ax.tick_params(labelbottom=False)
                pass
            if j == 0:
                plt.ylabel(titles[i].capitalize(),
                           rotation=0, ha='right', va='baseline')

            ax.tick_params(which='both', left=False, right=False,
                           labelleft=False, labelright=False, top=False,
                           labeltop=False)
            # ax.tick_params(which='minor', left=False, right=False,
            #                labelleft=False, labelright=False, top=False, labeltop=False)

        if i == 0:
            ax = plt.subplot(Nrows, 3, 3)
            ax.axis('off')
            ax.legend(pc, [f"{_damp:.4f}" for _damp in damping_list], loc='upper left',
                      fontsize='xx-small', title='Damping', frameon=False,
                      borderaxespad=0.0, title_fontsize='x-small')

    plt.subplots_adjust(hspace=0.075, wspace=0.125, left=0.075,
                        right=1.1, bottom=0.3, top=0.875)


def plot_results_M0(dms, damping_list, cat: CMTCatalog, titles=None):

    if isinstance(dms, list) is False:
        dms = [deepcopy(dms)]
    else:
        dms = deepcopy(dms)

    Nrows = len(dms)
    plt.figure(figsize=(10, 1 + Nrows*1))

    labels = ['$M_0$', '$t_{cmt}$', '$z$', '$\\theta$', '$\phi$']

    # Get normalizaton
    M0 = cat.getvals(vtype='M0')

    # Get the old moment tensor elements
    otensor = cat.getvals(vtype='tensor')
    print(otensor.shape)

    # Get colors
    colors = pick_colors_from_cmap(len(damping_list), 'rainbow')

    pdms = []
    for i in range(len(dms)):
        dtensor = dms[i][:, :, :6]
        print(dtensor.shape)
        tdms = dms[i][:, :, 5:]
        for j in range(tdms.shape[0]):
            tdms[j, :, 0] = (
                tensor2M0(otensor[:, :] + dtensor[j, :, :]) - M0)/M0

        pdms.append(tdms)

    dms = pdms

    for i in range(len(pdms)):

        # Get qunatiles of wides range
        dms[i][:, :, 2] = dms[i][:, :, 2]/1000.0

        xrange = (
            np.quantile(dms[i][0, :, :], 0.02, axis=0),
            np.quantile(dms[i][0, :, :], 0.98, axis=0)
        )

        for j in range(5):

            ax = plt.subplot(Nrows, 6, 6*i + (1+j))

            if i == 0:
                plt.title(labels[j])
            pc = []
            for k, _damp in enumerate(damping_list):

                if i == 0:
                    pc.append(
                        plt.plot([], [], label=f"{_damp}", c=colors[k])[0])

                plt.hist(
                    dms[i][k, :, j], bins=100, range=(xrange[0][j], xrange[1][j]),
                    histtype='stepfilled', edgecolor=colors[k],
                    facecolor='none', density=True)

            if i == len(dms)-1:
                ax.set_xticklabels(ax.get_xticks(), rotation=-45)
            else:
                ax.set_xticklabels([])

        if i == 0:
            ax = plt.subplot(Nrows, 6, 6)
            ax.axis('off')
            ax.legend(pc, [str(_damp) for _damp in damping_list], loc='upper right',
                      fontsize='xx-small', title='Damping', frameon=False,
                      borderaxespad=0.0, title_fontsize='x-small')

    plt.subplots_adjust(hspace=0.5, wspace=1.0, left=0.025,
                        right=0.975, bottom=0.2, top=0.8)


def update_catalog(H, G, scaling, cat: CMTCatalog) -> CMTCatalog:

    # Damping value
    damping = 0.01

    # Compute Trace for entire Array
    tr = np.trace(H, axis1=1, axis2=2)

    # Create damping array
    d = np.diag(np.ones(10))
    # d[:6, :6] = 0

    # Create scaling array
    s = scaling
    sdiag = np.diag(scaling)

    # Get Scalar moment of each event
    M0 = cat.getvals(vtype='M0')

    # Initialize model vectors
    dm = np.zeros_like(G)

    # Zero trace params
    zero_trace = True
    pos = np.arange(10)
    m, n = H[0, :, :].shape
    # Zero trace things
    if zero_trace:
        d = np.diag(np.ones(11))
        d[-1] = 0
        h = np.zeros((m+1, n+1))
        g = np.zeros(m+1)
        zero_trace_array = np.zeros(11)
        zero_trace_array[:3] = 1
        h[:, -1] = zero_trace_array
        h[-1, :] = zero_trace_array

        zp = slice(0, -1)
    else:
        d = np.diag(np.ones(10))
        h = np.zeros((m, n))
        g = np.zeros(m)
        zp = slice(0, m)

    for _i, (_h, _g, m0, _ev) in enumerate(zip(H, G, M0, cat)):
        # Fix scaling
        s[:6] = m0
        sdiag[pos[:6], pos[:6]] = m0

        h[zp, zp] = sdiag @ _h @ sdiag
        g[zp] = _g * s

        if zero_trace:
            g[-1] = np.sum(np.array([_ev.tensor[:3]]))/m0

        # Fix Hessians
        dm[_i, :] = np.linalg.solve(
            h + damping * tr[_i] * d, -g)[:10] * s

    # Don't change onld catalog, but create new one
    cat1 = deepcopy(cat)

    for _i, ev in enumerate(cat1):
        setattr(ev, 'm_rr', ev.m_rr + dm[_i, 0])
        setattr(ev, 'm_tt', ev.m_tt + dm[_i, 1])
        setattr(ev, 'm_pp', ev.m_pp + dm[_i, 2])
        setattr(ev, 'm_rt', ev.m_rt + dm[_i, 3])
        setattr(ev, 'm_rp', ev.m_rp + dm[_i, 4])
        setattr(ev, 'm_tp', ev.m_tp + dm[_i, 5])
        ev.time_shift = ev.time_shift + dm[_i, 6]
        ev.depth_in_m = ev.depth_in_m + dm[_i, 7]
        ev.latitude = ev.latitude + dm[_i, 8]
        ev.longitude = ev.longitude + dm[_i, 9]
        # ev.update_hdur()

    return cat1
