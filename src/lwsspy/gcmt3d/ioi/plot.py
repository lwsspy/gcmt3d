import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
from lwsspy.plot.axes_from_axes import axes_from_axes

# def cgh(costdir, gradir, hessdir, it, ls=None):
#     c = read_cost(costdir, it, ls)
#     g = read_gradient(graddir, it, ls)
#     H = read_hessian(hessdir, it, ls)

#     return c, g, H


def plot_cost(optdir):

    # Cost dir
    costdir = os.path.join(optdir, 'cost')

    clist = []
    for _cfile in sorted(os.listdir(costdir)):
        if "_ls00000.npy" in _cfile:
            clist.append(np.load(os.path.join(costdir, _cfile)))

    plt.figure()
    ax = plt.axes()
    plt.plot(np.log10(clist/clist[0]))
    plt.xlabel("Iteration #")
    plt.ylabel("$\\log_{10}\\,C/C_0$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show(block=False)


def plot_hessians(optdir):
    hessdir = os.path.join(optdir, 'hess')
    s = np.load(os.path.join(optdir, 'scaling.npy'))

    mlist = []
    for _mfile in sorted(os.listdir(hessdir)):
        if "_ls00000.npy" in _mfile:
            mlist.append(np.load(os.path.join(hessdir, _mfile)))

    N = len(mlist)
    n = int(np.ceil(np.sqrt(N)))

    # Get number of rows and colums
    ncols = n
    if N/n < n:
        nrows = n - 1
    else:
        nrows = n

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2*ncols + 1.0, nrows*2+1.0))
    plt.subplots_adjust(hspace=0.4)

    counter = 0
    for _i in range(nrows):

        for _j in range(ncols):
            if len(mlist) > counter:
                im = axes[_i][_j].imshow(
                    np.diag(s) @ mlist[counter] @ np.diag(s))
                axes[_i][_j].axis('equal')
                axes[_i][_j].axis('off')
                axes[_i][_j].set_title(f"{counter}")
                cax = axes_from_axes(
                    axes[_i][_j], 99080+counter, [0., -.05, 1.0, .05])
                plt.colorbar(im, cax=cax, orientation='horizontal')
            counter += 1
    plt.show(block=False)


def plot_model(optdir):

    modldir = os.path.join(optdir, 'modl')
    mlist = []
    for _mfile in sorted(os.listdir(modldir)):
        if "_ls00000.npy" in _mfile:
            mlist.append(np.load(os.path.join(modldir, _mfile)))

    mlist = np.array(mlist)
    plt.figure()
    ax = plt.axes()
    plt.plot(mlist/mlist[0])
    plt.xlabel("Iteration #")
    plt.ylabel("$M/M_0$")
    # ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend([_i for _i in range(mlist[0].size)])

    plt.show(block=False)


def plot_cm():
    clist = []
    for _cfile in sorted(os.listdir(costdir)):
        clist.append(np.load(os.path.join(costdir, _cfile)))

    N = 200
    marray = np.zeros((m.size, N))
    for _i in range(N):
        marray[:, _i] = read_model(modldir, _i, ls=0)

    mnorm = 0.5*np.sum((marray - m_sol[:, np.newaxis])**2, axis=0)/m.size

    ax = axes()
    plot(clist/clist[0])
    plot(mnorm/mnorm[0])
    # ax.set_yscale('log')


def plot_gh():
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
    axes[0].plot(g)
    im = axes[1].imshow(H)
    plt.colorbar(im)


def plot_g(mnames, it, ls):

    data = read_data_processed(datadir)
    synt = read_synt(syntdir, it, ls)
    dsdm = []
    for _i, (_key, _m) in enumerate(mdict.items()):
        dsdm.append(read_frechet(_i, frecdir, it, ls))

    plt.figure(figsize=(10, 6))

    # Plot the plain data
    ax = plt.subplot(251)
    ax.axis('off')
    plt.imshow(data)
    plt.text(0, 1, 'Data', ha='left',
             va='bottom', transform=ax.transAxes)

    # Plot the difference between model and data
    ax = plt.subplot(252)
    ax.axis('off')
    plt.imshow(data-synt)
    plt.text(0, 1, 'Difference', ha='left',
             va='bottom', transform=ax.transAxes)
    plt.text(0, 0, f'Misfit: {0.5/synt.size*np.sum((synt-data)**2):.4f}', ha='left',
             va='top', transform=ax.transAxes)

    for _i, _key in enumerate(mnames):
        ax = plt.subplot(252 + 1 + _i)
        ax.axis('off')
        plt.imshow(dsdm[_i])
        plt.text(0, 1, _key, ha='left',
                 va='bottom', transform=ax.transAxes)
