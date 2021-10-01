import os
import datetime
import pickle
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from obspy import Stream
from ...seismo.plot_seismogram import plot_seismogram_by_station
from ...seismo.source import CMTSource


def read_traces(wtypes, streamdir):

    # Initialize dictionary
    d = dict()

    # Populate dictionary
    for _wtype in wtypes:
        with open(os.path.join(streamdir, f"{_wtype}_stream.pkl"), 'rb') as f:
            if 'synthetic' in streamdir:
                d[_wtype] = dict(synt=pickle.load(f))
            else:
                d[_wtype] = pickle.load(f)

    return d


def unique_dist_sorted_stations(st: Stream):
    stations = set()

    for _tr in st:
        stations.add(
            (_tr.stats.distance, _tr.stats.network, _tr.stats.station))

    stations = list(stations)
    stations.sort()

    return [_st[1] for _st in stations], [_st[2] for _st in stations]


def plot_traces(
        cmtsource: CMTSource,
        data: dict,
        synt1: dict,
        synt2: Optional[dict] = None,
        outputdir="."):

    plt.switch_backend("pdf")

    if isinstance(synt2, dict) is False:
        synt2 = dict()
        for k in data.keys():
            synt2[k] = dict()
            synt2[k]["synt"] = None

    # Go through traces
    for _wtype in data.keys():

        print(f"Plotting {_wtype} waves")

        with PdfPages(os.path.join(outputdir, f"final_windows_{_wtype}.pdf")) as pdf:

            # Sorted the networks ad stations by distances of the traces
            networks, stations = unique_dist_sorted_stations(data[_wtype])

            # Check
            for _net, _sta in zip(networks, stations):
                try:
                    plot_seismogram_by_station(
                        _net, _sta,
                        data[_wtype],
                        synt1[_wtype]["synt"],
                        synt2[_wtype]["synt"],
                        cmtsource,
                        tag=_wtype)

                # synt_tr = self.synt1[_wtype]["synt"].select(
                #     station=obsd_tr.stats.station,
                #     network=obsd_tr.stats.network,
                #     component=obsd_tr.stats.component)[0]
                # if
                # init_synt_tr = self.synt_dict_init[_wtype]["synt"].select(
                #     station=obsd_tr.stats.station,
                #     network=obsd_tr.stats.network,
                #     component=obsd_tr.stats.component)[0]
                except Exception as err:
                    print(f"obsd trace({_net}.{_sta}): {err}")
                    continue

                # fig = plot_seismograms(
                #     obsd_tr, init_synt_tr, synt_tr, self.cmtsource,
                #     tag=_wtype)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

                # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
            d['Author'] = 'Lucas Sawade'
            d['Subject'] = 'Trace comparison in one pdf'
            d['Keywords'] = 'seismology, moment tensor inversion'
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
