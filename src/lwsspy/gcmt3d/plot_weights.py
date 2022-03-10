import _pickle as cPickle
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import lwsspy.plot as lplt
import lwsspy.maps as lmap
from cartopy.crs import PlateCarree, AzimuthalEquidistant, Mollweide
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
import numpy as np


def plot_single_weight_set(
        ax, lat, lon, weights, nomean=False, vmin=None, vmax=None,
        nocolorbar=False):

    lmap.plot_map()

    # Custom minmax for the colorbar
    if not vmin:
        vmin = np.min(weights)
    if not vmax:
        vmax = np.max(weights)

    # Plot nonnormalized weights
    if nomean:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "rainbow"
    else:
        norm = lplt.MidPointLogNorm(vmin=vmin, vmax=vmax, midpoint=1.0)
        cmap = "RdBu_r"

    plt.scatter(lon, lat, c=weights, cmap=cmap,
                norm=norm,
                edgecolors='k', linewidths=0.5,
                transform=PlateCarree())

    # formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    if nocolorbar is False:
        cb = lplt.nice_colorbar(
            orientation='horizontal', aspect=40, pad=0.075)
        # np.arange(0.3, 3.0, 0.3),
        #    ticks=[0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],
        #    format=formatter)
        cb.set_label("Weights")

    lplt.plot_label(
        ax,
        f"R = {np.max(weights)/np.min(weights):4.2f}",
        location=3, box=False, dist=0.0, fontdict=dict(fontsize='small'))

    lplt.plot_label(
        ax, f"S = {np.sum(weights):4.2f}",
        location=4, box=False, dist=0.0, fontdict=dict(fontsize='small'))


def plot_single_weight_hist(ax, weights, nbins=10, color="lightgray"):

    weights_norm = weights/np.min(weights)
    nb, _, _ = plt.hist(weights_norm,
                        bins=nbins,
                        edgecolor='k',
                        facecolor=color,
                        linewidth=0.75,
                        linestyle='-',
                        histtype='stepfilled')

    lplt.plot_label(
        ax,
        f"min: {np.min(weights_norm):7.4f}\n"
        f"max: {np.max(weights_norm):7.4f}\n"
        f"sum: {np.sum(weights_norm):7.4f}\n"
        f"mean: {np.mean(weights_norm):7.4f}\n"
        f"median: {np.median(weights_norm):7.4f}",
        location=7, box=True, dist=-0.1, fontdict=dict(fontsize='small'))


def plot_weightpickle(weightpickle: str):

    with open(weightpickle, "rb") as f:
        weights = cPickle.load(f)

    plot_weights(weights)


def plot_weight_histograms(weights: dict):

    # Weights to be plotted
    component_list = ["Z", "R", "T"]
    weightlist = ["geographical", "azimuthal", "combination", "final"]

    for _j, _weight in enumerate(weightlist):

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig)
        plt.subplots_adjust(bottom=0.075, top=0.925, left=0.075, right=0.925)

        counter = 0
        for _i, _wtype in enumerate(weights.keys()):
            if _wtype == "event":
                continue

            # Get wave weight
            waveweight = weights[_wtype]["weight"]

            for _j, _component in enumerate(component_list):

                # Create axes
                ax = plt.subplot(gs[counter, _j])

                # Get weights
                plotweights = weights[_wtype][_component][_weight]

                # Plot histogram
                plot_single_weight_hist(ax, plotweights, nbins=10)

                if _j == 0:
                    # lplt.plot_label(ax, _wtype.capitalize() + f": {waveweight}",
                    #                location=14, box=False, dist=0.05)

                    plt.ylabel(_wtype.capitalize() + f": {waveweight:4.2f}")
                if counter == 2:
                    # lplt.plot_label(ax, _component.capitalize(),
                    #                location=13, box=False, dist=0.05)
                    plt.xlabel(_component.capitalize())

            counter += 1

        plt.suptitle(f"{_weight.capitalize()}")
        plt.savefig(f"./weights_{_weight}_histogram.pdf")


def plot_weight_histogram_pickle(weightpickle: str):

    with open(weightpickle, "rb") as f:
        weights = cPickle.load(f)

    plot_weight_histograms(weights)


def plot_weights(weights: dict):

    # Weights to be plotted
    component_list = ["Z", "R", "T"]
    weightlist = ["geographical", "azimuthal", "combination", "final"]

    # Event location
    lat0, lon0 = weights["event"]

    for _wtype in weights.keys():
        if _wtype == "event":
            continue
        # Get wave weight
        waveweight = weights[_wtype]["weight"]

        # Create Base figure
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, len(weightlist), figure=fig)
        plt.subplots_adjust(bottom=0.025, top=0.925, left=0.05, right=0.95)

        for _i, _component in enumerate(component_list):
            latitudes = weights[_wtype][_component]["lat"]
            longitudes = weights[_wtype][_component]["lon"]

            for _j, _weight in enumerate(weightlist):

                # Create axes
                ax = plt.subplot(gs[_i, _j], projection=Mollweide(
                    central_longitude=lon0))
                ax.set_global()

                # Get weights
                plotweights = weights[_wtype][_component][_weight]

                # Plot weights
                if _weight == "final":
                    nomean = True
                else:
                    nomean = False

                plot_single_weight_set(
                    ax, latitudes, longitudes, plotweights, nomean=nomean)

                if _i == 0:
                    lplt.plot_label(ax, _weight.capitalize(),
                                        location=14, box=False, dist=0.05)
                if _j == 0:
                    lplt.plot_label(ax, _component.capitalize(),
                                        location=13, box=False, dist=0.05)

        plt.suptitle(f"{_wtype.capitalize()}: {waveweight:6.4f}")
        plt.savefig(f"./weights_{_wtype}.pdf")


def plot_final_weights(weights: dict):

    # Weights to be plotted
    component_list = ["Z", "R", "T"]

    # Event location
    lat0, lon0 = weights["event"]

    # Create Base figure
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(3, 3, figure=fig)
    plt.subplots_adjust(bottom=0.1, top=0.925, left=0.05,
                        right=0.95, hspace=0.025)

    neg = 0
    allweights = []
    for _j, _wtype in enumerate(weights.keys()):
        if _wtype == "event":
            neg += 1
            continue
        # Get wave weight
        waveweight = weights[_wtype]["weight"]

        for _i, _component in enumerate(component_list):

            allweights.extend(
                np.array(weights[_wtype][_component]["final"]) * waveweight)

    vmin = np.min(allweights)
    vmax = np.max(allweights)

    neg = 0
    for _j, _wtype in enumerate(weights.keys()):
        if _wtype == "event":
            neg += 1
            continue

        # Get wave weight
        waveweight = weights[_wtype]["weight"]

        for _i, _component in enumerate(component_list):
            latitudes = weights[_wtype][_component]["lat"]
            longitudes = weights[_wtype][_component]["lon"]

            # Create axes
            ax = plt.subplot(gs[_i, _j-neg], projection=Mollweide(
                central_longitude=lon0))
            ax.set_global()

            # Plot event
            plt.plot(lon0, lat0, '*k', markersize=10, transform=PlateCarree())

            # Get weights
            plotweights = np.array(weights[_wtype][_component]["final"])

            # Plot weight
            plot_single_weight_set(
                ax, latitudes, longitudes, plotweights * waveweight,
                nomean=True, vmin=vmin, vmax=vmax, nocolorbar=True)

            if _i == 0:
                lplt.plot_label(ax, f"{_wtype.capitalize()}: {waveweight:6.4f}",
                                    location=14, box=False, dist=0.05)
            if _j - neg == 0:
                lplt.plot_label(ax, _component.capitalize(),
                                    location=13, box=False, dist=0.05)

            if _j-neg == 1 and _i == 2:
                cax = lplt.axes_from_axes(
                    ax, n=91230, extent=[-0.25, -0.25, 1.5, 0.05])
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = "rainbow"

                sc = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
                cb = lplt.nice_colorbar(
                    sc, cax=cax, orientation='horizontal', aspect=40, pad=0.075)
                cb.set_label("Weights")

            if _i == 2 and _j-neg == 0:
                lplt.plot_label(ax, f"$\mathrm{{R}}_\mathrm{{T}}$ = {vmax/vmin:4.2f}",
                                    location=11, box=False, dist=0.125)
            if _i == 2 and _j-neg == 2:
                lplt.plot_label(ax, f"$\mathrm{{S}}_\mathrm{{T}}$ = {np.sum(allweights):4.2f}",
                                    location=10, box=False, dist=0.125)

        # plt.suptitle(f"{_wtype.capitalize()}: {waveweight:6.4f}")
    plt.savefig(f"./weights_final.pdf")


def plot_final_weight_pickle(weightpickle: str):

    with open(weightpickle, "rb") as f:
        weights = cPickle.load(f)

    plot_final_weights(weights)
