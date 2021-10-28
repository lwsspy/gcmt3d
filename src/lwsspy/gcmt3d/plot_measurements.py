# External
import _pickle as cPickle
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from copy import deepcopy
from glob import glob
from typing import Optional
import os
# Internal
from lwsspy import plot as lplt
from lwsspy import base as lbase
from lwsspy import seismo as lseis
from lwsspy import math as lmat

lplt.updaterc()


def get_bins(b, a, nbin, mtype):

    if mtype == "max_cc":
        ax_min = np.min((np.min(b), np.min(a)))
        ax_max = np.max((np.max(b), np.max(a)))
    if mtype == "dlna":
        ax_min = -1  # np.min((np.min(b), np.min(a)))
        ax_max = 1  # np.max((np.max(b), np.max(a)))
    elif mtype == "chi":
        ax_min = 0.0
        ax_max = 2.0  # np.max((np.max(b), np.max(a)))
    elif mtype == "misfit":
        ax_min = 0.0
        ax_max = 2.0  # np.max((np.max(b), np.max(a)))
    elif mtype == "time_shift":
        ax_min = -20.0
        ax_max = 20.0  # np.max((np.max(b), np.max(a)))
    else:
        ax_min = np.min((np.min(b), np.min(a)))
        ax_max = np.max((np.max(b), np.max(a)))
        abs_max = np.max((np.abs(ax_min), np.abs(ax_max)))
        ax_min = -abs_max
        ax_max = abs_max
    binwidth = (ax_max - ax_min) / nbin

    return np.arange(ax_min, ax_max + binwidth / 2., binwidth)


def plot_measurement_pkl(
        measurement_pickle_before: str,
        measurement_pickle_after: str,
        alabel: Optional[str] = None,
        blabel: Optional[str] = None,
        mtype="chi", no_after: bool = False):

    with open(measurement_pickle_before, "rb") as f:
        measurements_before = cPickle.load(f)
    with open(measurement_pickle_after, "rb") as f:
        measurements_after = cPickle.load(f)

    plot_measurements(
        measurements_before, measurements_after, mtype=mtype,
        blabel=blabel, alabel=alabel, no_after=no_after)


def get_measurement(mdict: dict, mtype: str):

    if mtype == "chi":
        # Get the data type from the measurement dictionary
        m = np.array(mdict["dL2"])/np.array(mdict["L2"])

    elif mtype == "misfit":
        # Get the data type from the measurement dictionary
        m = np.array(mdict["dL2"])/np.array(mdict["trace_energy"])

    else:
        m = np.array(mdict[mtype])

    return m


def plot_measurements(before: dict, after: dict, alabel: Optional[str] = None,
                      blabel: Optional[str] = None, mtype='chi',
                      no_after: bool = False, leftlabel=True):

    # Get number of wave types:
    Nwaves = len(before.keys())

    # Create base figure
    fig = plt.figure(figsize=(6.5, 0.5+Nwaves*1.5))
    gs = GridSpec(Nwaves, 3, figure=fig)
    # plt.subplots_adjust(bottom=0.075, top=0.95,
    #                     left=0.05, right=0.95, hspace=0.25)
    plt.subplots_adjust(bottom=0.1, top=0.9,
                        left=0.125, right=0.975,
                        hspace=0.4, wspace=0.35)

    # Create subplots
    counter = 0
    components = ["Z", "R", "T"]

    # Get the amount of colors
    # colors = pick_colors_from_cmap(len(components), cmap='rainbow')
    colors = np.array([[0.8, 0, 0, 1], [0, 0.8, 0, 1], [0, 0, 0.8, 1]])

    if mtype == "time_shift":
        component_bins = [21, 21, 21]
    else:
        component_bins = [75, 75, 75]

    # Plot centerline for orientation if dist is similar to gaussian..
    plotcenterline = True if mtype in ["time_shift", "dlna"] else False

    if blabel is None:
        blabel = "$m_0$"

    if alabel is None:
        alabel = "$m_f$"

    for _i, (_wtype, _compdict) in enumerate(before.items()):
        for _j, (_comp, _bins) in enumerate(zip(components, component_bins)):

            bdict = _compdict[_comp]
            adict = after[_wtype][_comp]

            a = get_measurement(adict, mtype)
            b = get_measurement(bdict, mtype)

            # Set alpha color
            acolor = deepcopy(colors[_j, :])
            acolor[3] = 0.5

            # Create plot
            ax = plt.subplot(gs[_i, _j])

            # Plot before
            if no_after is False:
                blinestyle = "-"
                lcolor = 'none'
                fcolor = 'lightgray'

            else:
                blinestyle = "-"
                lcolor = colors[_j, :]
                fcolor = 'none'

            bins = get_bins(b, a, _bins, mtype)
            nb, _, _ = plt.hist(b,
                                bins=bins,
                                edgecolor=lcolor,
                                facecolor=fcolor, linewidth=0.5,
                                linestyle=blinestyle,
                                histtype='stepfilled',
                                density=True)
            plt.plot([], [], color=colors[_j, :],
                     linewidth=0.75, linestyle=blinestyle, label=blabel)

            # Plot After
            if no_after is False:
                na, _, _ = plt.hist(a,
                                    bins=bins,
                                    edgecolor=colors[_j, :],
                                    facecolor='none', linewidth=1.25,
                                    linestyle="-",
                                    histtype='step',
                                    density=True)
                plt.plot([], [], color=colors[_j, :],
                         linewidth=0.75, linestyle="-", label=alabel)

            # Annotations
            if no_after is False:
                mlabel = int(np.max((len(alabel), len(blabel))))
                label = f"{blabel:<{mlabel}}: {np.mean(b):5.2f}±{np.std(b):4.2f}\n"
                label += f"{alabel:<{mlabel}}: {np.mean(a):5.2f}±{np.std(a):4.2f}"
            else:
                label = f"{blabel}: {np.mean(b):5.2f}±{np.std(b):4.2f}"
            lplt.plot_label(ax, label, location=2, box=False, fontfamily="monospace",
                            fontsize="x-small", dist=0.025)
            lplt.plot_label(ax, f"N: {len(b)}", location=7, box=False,
                            fontsize="small", dist=0.025)
            lplt.plot_label(ax, lbase.abc[counter], location=6, box=False,
                            fontsize="small", dist=0.025)

            if no_after is False:
                nmax = np.max([np.max(nb), np.max(na)])
            else:
                nmax = np.max(nb)

            if plotcenterline:
                plt.plot([0, 0], [0, 1.1*nmax], "k--", lw=1.00)
                plt.plot([0, 0], [1.1*nmax, 1.5*nmax], ":",
                         lw=1.00, c='lightgray', zorder=-1)

            ax.set_ylim((0, 1.5*nmax))

            if mtype == "chi":
                location = 'upper left'
                ncol = 1
            else:
                location = 'upper left'
                ncol = 1

            if no_after is False:
                pass
                # plt.legend(loc=location, fontsize='x-small',
                #            fancybox=False, frameon=False,
                #            ncol=ncol, borderaxespad=0.0, borderpad=0.5,
                #            handletextpad=0.15, labelspacing=0.0,
                #            handlelength=1.0, columnspacing=1.0)
            # lplt.plot_label(ax, lbase.abc[counter] + ")", location=6, box=False,
            #                fontsize="small")

            # if _wtype == "body" and _comp == "Z":
            #     ax.set_xlim((-0.001, 0.1))

            if _j == 0 and leftlabel is True:
                plt.ylabel(_wtype.capitalize())

            if _i == Nwaves-1:
                plt.xlabel(_comp.capitalize())
            else:
                ax.tick_params(labelbottom=False)
            counter += 1
    return fig
    # plt.show(block=False)


def get_illumination(mat: np.ndarray, minillum: int = 25,
                     illumdecay: int = 50, r: bool = False):
    # Get locations of the numbers
    below_minillum = np.where(mat < minillum)
    between = np.where((minillum < mat) & (mat < illumdecay))
    above_illumdecay = np.where(mat > illumdecay)

    # Create alpha array
    alphas = np.zeros_like(mat)
    alphas[above_illumdecay] = (illumdecay - minillum)
    alphas[below_minillum] = 0
    alphas[between] = mat[between] - minillum
    alphas = alphas/(illumdecay - minillum)

    if r:
        alphas[between] = 1.0 - alphas[between]
        alphas[above_illumdecay] = 0.0
        alphas[below_minillum] = 1.0
    return alphas


def plot_window_histograms(xbins, ybins, hists: dict):

    dpd = dict(
        body=dict(
            counts=dict(
                cmap="gray_r",
                # norm=colors.LogNorm(vmin=1, vmax=2000),
                norm=colors.Normalize(vmin=0, vmax=1500),
                label="Counts"),
            dlna=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(midpoint=0.0),
                label="dlnA"),
            time_shift=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(
                    vmin=-12.5, vmax=12.5, midpoint=0.0),
                label="CC-$\\Delta$ t"),
            maxcc=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.9, vmax=0.98),
                label="Max. CrossCorr"),
            chi=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.0, vmax=1.0),
                label="Chi")
        ),
        surface=dict(
            counts=dict(
                cmap="gray_r",
                # norm=colors.LogNorm(vmin=100, vmax=10000),
                norm=colors.Normalize(vmin=0, vmax=2000),
                label="Counts"),
            dlna=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(midpoint=0.0),
                label="dlnA"),
            time_shift=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(
                    vmin=-12.5, vmax=12.5, midpoint=0.0),
                label="CC-$\\Delta$ t"),
            maxcc=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.9, vmax=0.98),
                label="Max. CrossCorr"),
            chi=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.0, vmax=1.0),
                label="Chi")
        ),
        mantle=dict(
            counts=dict(
                cmap="gray_r",
                # norm=colors.LogNorm(vmin=100, vmax=10000),
                norm=colors.Normalize(vmin=0, vmax=2000),
                label="Counts"),
            dlna=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(midpoint=0.0),
                label="dlna"),
            time_shift=dict(
                cmap="seismic",
                norm=lplt.MidpointNormalize(
                    vmin=-12.5, vmax=12.5, midpoint=0.0),
                label="CC-$\\Delta$ t"),
            maxcc=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.9, vmax=0.98),
                label="Max. CrossCorr"),
            chi=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.0, vmax=1.0),
                label="Chi"))
    )

    # Compute ex
    xmin, xmax = np.min(xbins), np.max(xbins)
    ymin, ymax = np.min(ybins), np.max(ybins)
    extent = [xmin, xmax, ymin/60, ymax/60]

    # ["counts", "time_shift", "maxcc", "dlna", "chi"]:
    for _dat in ["counts"]:
        for _wtype in hists.keys():
            fig = plt.figure(figsize=(10, 6))
            fig.subplots_adjust(
                left=0.06, right=0.875, top=0.925, bottom=-0.05, wspace=0.2
            )
            spcount = 131
            axes = []
            for _i, _comp in enumerate(["Z", "R", "T"]):
                boolcounts = hists[_wtype][_comp]["counts"].astype(bool)
                # alphas = get_illumination(
                #     hists[_wtype][_comp]["counts"].T[::-1, :], 25, 75)
                alphas = 1

                # Define Data
                if _dat == "counts":
                    zz = hists[_wtype][_comp]["counts"]
                else:
                    zz = np.zeros_like(hists[_wtype][_comp][_dat])
                    zz[boolcounts] = \
                        hists[_wtype][_comp][_dat][boolcounts] / \
                        hists[_wtype][_comp]["counts"][boolcounts]

                if _i == 0:
                    axes.append(plt.subplot(spcount))
                    plt.ylabel("Traveltime [min]")
                else:
                    axes.append(plt.subplot(
                        spcount + _i, sharey=axes[0], sharex=axes[0]))
                    axes[_i].tick_params(labelleft=False)

                if _dat == "counts":
                    im1 = axes[_i].imshow(
                        zz.T[::-1, :], cmap=dpd[_wtype][_dat]['cmap'],
                        interpolation='none', extent=extent,
                        norm=dpd[_wtype][_dat]['norm'], aspect='auto',
                        zorder=-10)
                else:
                    im1 = axes[_i].imshow(
                        zz.T[::-1, :], cmap=dpd[_wtype][_dat]['cmap'],
                        interpolation='none',
                        norm=dpd[_wtype][_dat]['norm'],
                        extent=extent, aspect='auto', alpha=alphas,
                        zorder=-10)

                # Plot traveltimes and fix limits
                if _wtype == "body":
                    if _dat in ["dlna", "time_shift"]:
                        cmap = 'Dark2'
                    else:
                        cmap = 'rainbow'

                    axes[_i].set_ylim(0.0, 60.0)
                    super_phase_list = [
                        "P", "PP", "PPP", "PcP", "Pdiff", "PKP", "PS", "PPS",
                        "S", "SS", "SSS", "ScS", "Sdiff", "SKS", "SP", "SSP",
                        "SSSS", "ScSScS",
                    ]
                    cols = lplt.pick_colors_from_cmap(
                        len(super_phase_list), cmap)
                    colordict = {ph: col for ph,
                                 col in zip(super_phase_list, cols)}
                    # Custom lines for legend
                    custom_lines = [
                        Line2D([0], [0], color=col, lw=1, label=ph)
                        for ph, col in zip(super_phase_list, cols)]

                    if _comp not in ["R", "Z"]:
                        phase_list = [
                            "S", "SS", "SSS", "SSSS", "ScS", "Sdiff",
                            "ScSScS",
                        ]
                        plt.legend(
                            custom_lines, super_phase_list, loc="upper left",
                            bbox_to_anchor=[1, 1], frameon=False)
                    else:
                        phase_list = super_phase_list

                    lseis.plot_traveltimes(0, phase_list=phase_list, cmap=cmap,
                                           colordict=colordict, legend=False,
                                           markersize=0.5)

                elif _wtype == "surface":
                    axes[_i].set_ylim(0.0, 60.0)
                    # axes[_i].set_ylim(0.0, 120.0)
                else:
                    axes[_i].set_ylim(0.0, 60.0)
                    # axes[_i].set_ylim(0.0, 180.0)

                mu = zz[boolcounts].mean()
                sig = zz[boolcounts].std()
                labint, ndec = lplt.get_stats_label_length(mu, sig, ndec=2)
                lplt.plot_label(axes[_i],
                                f"$\\mu$ = {mu:>{labint}.{ndec}f}\n"
                                f"$\\sigma$ = {sig:>{labint}.2f}",
                                location=4, box=False,
                                fontdict=dict(
                    fontfamily="monospace",
                    fontsize="small"))

                # Plot Type label
                if _wtype == "mantle":
                    location = 2
                else:
                    location = 1
                lplt.plot_label(
                    axes[_i],
                    f"{_wtype.capitalize()}\n{_comp.capitalize()}",
                    location=location, box=False,
                    fontdict=dict(fontsize="small"))

                # Plot colorbar
                c = lplt.nice_colorbar(
                    matplotlib.cm.ScalarMappable(
                        cmap=im1.cmap, norm=im1.norm),
                    pad=0.025, orientation='horizontal', aspect=40)

                # Set labels
                c.set_label(dpd[_wtype][_dat]['label'])
                plt.xlabel("$\\Delta$ [$^\\circ$]")

                # Put xlabel on top
                axes[_i].xaxis.set_label_position('top')
                axes[_i].tick_params(labelbottom=False, labeltop=True)

                # Set rasterization zorder to rasteriz images for pdf
                # output
                axes[_i].set_rasterization_zorder(-5)


def compute_window_hists(mdict: dict, t_res: float = 5, deg_res: float = 1):

    hists = dict()
    components = ["Z", "R", "T"]

    xbins = np.arange(0, 181, deg_res)
    ybins = np.arange(0, 3600, t_res)

    for _i, _wtype in enumerate(mdict.keys()):

        hists[_wtype] = dict()

        for _j, _comp in enumerate(components):

            hists[_wtype][_comp] = dict()

            # Getting the neccessary measurements
            t0 = get_measurement(mdict[_wtype][_comp], "starttime")
            tf = get_measurement(mdict[_wtype][_comp], "endtime")
            dt = get_measurement(mdict[_wtype][_comp], "dt")
            epi = get_measurement(mdict[_wtype][_comp], "distance")
            time_shift = get_measurement(mdict[_wtype][_comp], "time_shift")
            maxcc = get_measurement(mdict[_wtype][_comp], "max_cc_calue")
            dlna = get_measurement(mdict[_wtype][_comp], "dlna")
            chi = get_measurement(mdict[_wtype][_comp], "chi")

            # Computing the vectors for each measurement
            # (this will probably take a while)
            t, e, tshift, mcc, amp, c = [], [], [], [], [], []

            N = len(t0)
            Nmag = int(lmat.magnitude(N)+1)
            for _k, (_t0, _tf, _dt, _epi, _time_shift, _maxcc, _dlna, _chi) in \
                    enumerate(zip(t0, tf, dt, epi, time_shift, maxcc, dlna, chi)):

                if (_k % 5000) == 0:
                    print(f"Window: {_k:{Nmag}}/{N}")
                t = np.arange(_t0, _tf, _dt)
                e = _epi * np.ones_like(t)
                tshift = _time_shift * np.ones_like(t)
                mcc = _maxcc * np.ones_like(t)
                amp = _dlna * np.ones_like(t)
                c = _chi * np.ones_like(t)

                counts, _, _ = np.histogram2d(
                    e, t, bins=(xbins, ybins))
                histtshift, _, _ = np.histogram2d(
                    e, t, bins=(xbins, ybins), weights=tshift)
                histmaxcc, _, _ = np.histogram2d(
                    e, t, bins=(xbins, ybins), weights=mcc)
                histdlna, _, _ = np.histogram2d(
                    e, t, bins=(xbins, ybins), weights=amp)
                histchi, _, _ = np.histogram2d(
                    e, t, bins=(xbins, ybins), weights=c)

                if _k == 0:
                    hists[_wtype][_comp]["counts"] = counts
                    hists[_wtype][_comp]["time_shift"] = histtshift
                    hists[_wtype][_comp]["maxcc"] = histmaxcc
                    hists[_wtype][_comp]["dlna"] = histdlna
                    hists[_wtype][_comp]["chi"] = histchi
                else:
                    hists[_wtype][_comp]["counts"] += counts
                    hists[_wtype][_comp]["time_shift"] += histtshift
                    hists[_wtype][_comp]["maxcc"] += histmaxcc
                    hists[_wtype][_comp]["dlna"] += histdlna
                    hists[_wtype][_comp]["chi"] += histchi

    return xbins, ybins, hists


def get_database_measurements(
        database: str, outdir: Optional[str] = None,
        blabel: Optional[str] = None, alabel: Optional[str] = None):

    if blabel is None:
        blabel = "before"

    if alabel is None:
        alabel = "after"

    # Get all directories
    cmtlocs = glob(os.path.join(database, '*/measurements*'))
    cmtlocs = list(set([os.path.dirname(cmtloc) for cmtloc in cmtlocs]))
    cmtlocs.sort()

    # Empty measurement lists
    components = ["Z", "R", "T"]
    for _cmtloc in cmtlocs:
        print(_cmtloc)
        try:
            measurement_pickle_before = os.path.join(
                _cmtloc, f"measurements_{blabel}.pkl"
            )
            measurement_pickle_after = os.path.join(
                _cmtloc, f"measurements_{alabel}.pkl"
            )
            print(measurement_pickle_before)
            print(measurement_pickle_after)
            with open(measurement_pickle_before, "rb") as f:
                measurements_before = cPickle.load(f)
            with open(measurement_pickle_after, "rb") as f:
                measurements_after = cPickle.load(f)

        except Exception as e:
            print(e)
            continue

        if "after" not in locals():
            before = measurements_before
            after = measurements_after

        else:

            for _wtype in measurements_before.keys():

                # Excape if the first dictionaries didn't have all the waves
                if _wtype not in before:
                    before[_wtype] = dict()
                    before[_wtype].update(measurements_before[_wtype])
                if _wtype not in after:
                    after[_wtype] = dict()
                    after[_wtype].update(measurements_after[_wtype])

                for _comp in components:
                    for _mtype in before[_wtype][_comp].keys():

                        # Grab
                        bdict = measurements_before[_wtype][_comp]
                        adict = measurements_after[_wtype][_comp]

                        # Get measurements
                        try:
                            b = get_measurement(bdict, _mtype)
                            a = get_measurement(adict, _mtype)

                            # Add to first dictionary
                            before[_wtype][_comp][_mtype].extend(b)
                            after[_wtype][_comp][_mtype].extend(a)
                        except KeyError:
                            print(f"Key Error for 'corr_ratio' at: {_cmtloc}")

    if outdir is not None:

        measurement_pickle_before_out = os.path.join(
            outdir, f"database_measurement_{blabel}.pkl"
        )
        measurement_pickle_after_out = os.path.join(
            outdir, f"database_measurement_{alabel}.pkl"
        )

        with open(measurement_pickle_before_out, "wb") as f:
            cPickle.dump(before, f)

        with open(measurement_pickle_after_out, "wb") as f:
            cPickle.dump(after, f)

    return before, after


def bin():

    import argparse
    lplt.updaterc()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', dest='database',
                        help='Database directory',
                        type=str, required=True)
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='Plot output directory',
                        required=True, type=str)
    parser.add_argument('-m', '--measurement', dest='measure', nargs='+',
                        type=str, default='chi')
    parser.add_argument('-a', '--alabel', dest='alabel',
                        type=str, default='after')
    parser.add_argument('-b', '--blabel', dest='blabel',
                        type=str, default='before')
    parser.add_argument('-na', '--no-after', dest='no_after',
                        help='Plot only the before dataset',
                        action='store_true', default=False)

    args = parser.parse_args()

    # Get the measurements
    before, after = get_database_measurements(
        args.database, alabel=args.alabel, blabel=args.blabel,
        outdir=args.outdir)

    if type(args.measure) is str:
        measure = [args.measure]
    else:
        measure = args.measure

    if args.outdir is not None:
        backend = plt.get_backend()
        plt.switch_backend('pdf')

    # Plot the measurements
    for _m in measure:
        plot_measurements(before, after, args.alabel,
                          args.blabel, mtype=_m, no_after=args.no_after)

        if args.outdir is not None:
            outfile = os.path.join(args.outdir, f"histograms_{_m}.pdf")
            plt.savefig(outfile, format='pdf')

    if args.outdir is not None:
        plt.switch_backend(backend)


def bin_plot_pickles():

    import argparse
    lplt.updaterc()

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='before',
                        help='Measurement before pickle',
                        type=str)
    parser.add_argument(dest='after',
                        help='Measurement after pickle',
                        type=str)
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='Plot output directory', default=None,
                        required=False, type=str)
    parser.add_argument('-m', '--measurement', dest='measure', nargs='+',
                        type=str, default='chi')
    parser.add_argument('-a', '--alabel', dest='alabel',
                        type=str, default='after')
    parser.add_argument('-b', '--blabel', dest='blabel',
                        type=str, default='before')
    parser.add_argument('-na', '--no-after', dest='no_after',
                        help='Plot only the before dataset',
                        action='store_true', default=False)

    args = parser.parse_args()

    if type(args.measure) is str:
        measure = [args.measure]
    else:
        measure = args.measure

    if args.outdir is not None:
        backend = plt.get_backend()
        plt.switch_backend('pdf')

    # Plot the measurements
    for _m in measure:
        plot_measurement_pkl(args.before, args.after, args.alabel,
                             args.blabel, mtype=_m, no_after=args.no_after)

        if args.outdir is not None:
            outfile = os.path.join(args.outdir, f"histograms_{_m}.pdf")
            plt.savefig(outfile, format='pdf')

    if args.outdir is not None:
        plt.switch_backend(backend)
