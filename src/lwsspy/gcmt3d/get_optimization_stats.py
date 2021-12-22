from math import e
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd


def get_files(databases: List[str], verbose: bool = True):

    # Get all cmts in database
    cmts = os.listdir(databases[0])

    filetable = []
    cmtf = []
    for cmt in cmts:
        row = []

        try:
            # For each database
            for _db in databases:

                # See whether
                file = os.path.join(_db, cmt, 'summary.npz')

                # The summary file exists
                if os.path.exists(file):

                    # Then append it to the row
                    row.append(file)

                else:
                    raise FileNotFoundError

        except FileNotFoundError as e:

            if verbose:
                print(f"{cmt:13s} not found in {_db}")

        filetable.append(row)
        cmtf.append(row)

    return cmtf, filetable


def get_optimization_stats(database):

    # Get all cmts in database
    cmts = os.listdir(database)

    summaryfiles = []
    for cmt in cmts:

        # Define summary filename
        summaryfile = os.path.join(database, cmt, 'summary.npz')

        # Check if exists and append if does
        if os.path.exists(summaryfile):
            summaryfiles.append(summaryfile)

    print("Number if inversions availables:", len(summaryfiles))

    # Summary
    iterations = []

    for summaryfile in summaryfiles:
        summary = np.load(summaryfile)
        iterations.append(len(summary['fcost_hist'][1:]))

    print("Average number of iterations:", np.mean(iterations))

    # Plot
    bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ticks = bins[:-1] + np.diff(bins)/2

    plt.figure(figsize=(4, 3))
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)
    ax = plt.gca()
    plt.hist(iterations, bins=bins, rwidth=0.8, color=(0.3, 0.3, 0.8))
    plt.xticks(ticks)
    ax.set_yscale('log')
    ax.set_xlabel("# of iterations")
    ax.set_ylabel("N")
    plt.savefig("iterationstats.pdf")
    # plt.show()


def bin():
    plt.switch_backend('pdf')
    import argparse
    import lwsspy.plot as lplt
    lplt.updaterc()

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='data_database', type=str,
        help='Database that contains the downloaded data.')
    args = parser.parse_args()

    get_optimization_stats(args.data_database)
