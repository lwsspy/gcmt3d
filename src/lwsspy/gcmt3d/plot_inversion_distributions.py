
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


def inversion_tests(events, G, H, scaling, cat, hypo_only=False):

    # Damping list
    damping_list = np.logspace(-4, -2, 7)

    # Initialize an array for model changes for all events and all dampings
    dm = np.zeros((len(damping_list), *G.shape))

    # Initalize M0 array
    M0 = np.zeros(G.shape[0])

    # Scaling array
    scale = np.ones_like(G)

    # Damping diagonal matrix
    if hypo_only:
        dampd = np.diag(np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0]))
    else:
        dampd = np.diag(np.ones(10))

    for _i, _cmt in enumerate(cat):

        M0[_i] = _cmt.M0
        scale[_i, :] = scaling
        scale[_i, :6] = M0[_i]
        g = G[_i, :] * scale[_i, :]
        h = np.diag(scale[_i, :]) @ H[_i, :, :] @ np.diag(scale[_i, :])

        for _j, _damp in enumerate(damping_list):
            print(
                f"{100*(_i+1)/G.shape[0]:3.0f}% -- {events[_i]:16} -- {_damp:10f}", end='\r')
            dm[_j, _i, :] = np.linalg.solve(
                h + _damp * np.trace(h) * dampd,
                -g)

    dms = dm * scale

    return dms, damping_list


def plot_results(dms, damping_list, cat: CMTCatalog, titles=None):

    if isinstance(dms, list) is False:
        dms = [deepcopy(dms)]
    else:
        dms = deepcopy(dms)

    Nrows = len(dms)
    plt.figure(figsize=(12, 1 + Nrows*1.25))

    labels = ['$M_{rr}$', '$M_{tt}$', '$M_{pp}$', '$M_{rt}$',
              '$M_{rp}$', '$M_{tp}$', '$t_{cmt}$', '$z$', '$\\theta$', '$\phi$']

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
        dms[i][:, :, 7] = dms[i][:, :, 7]/1000.0

    xrange = (
        np.quantile(dms[0][0, :, :], 0.02, axis=0),
        np.quantile(dms[0][0, :, :], 0.98, axis=0)
    )

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
                    dms[i][k, :, j], bins=100, range=(xrange[0][j], xrange[1][j]),
                    histtype='stepfilled', edgecolor=colors[k],
                    facecolor='none', density=True)

            if i == len(dms)-1:
                ax.set_xticklabels(ax.get_xticks(), rotation=-45)
            else:
                ax.set_xticklabels([])

        if i == 0:
            ax = plt.subplot(Nrows, 11, 11)
            ax.axis('off')
            ax.legend(pc, [f"{_damp:.4f}" for _damp in damping_list], loc='upper right',
                      fontsize='xx-small', title='Damping', frameon=False,
                      borderaxespad=0.0, title_fontsize='x-small')

    plt.subplots_adjust(hspace=0.2, wspace=1.0, left=0.025,
                        right=0.975, bottom=0.2, top=0.8)


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
