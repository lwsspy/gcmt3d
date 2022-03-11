"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import argv, exit
from ..functions.processing import process_data


def bin():
    """

    Usage:

        gcmt3d-process-data eventdir

    This script calls a python function that process the data and saves them into
    an event file.

    """

    # Get args or print usage statement 
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        outdir = argv[1]

    # Run the initializer
    process_data(outdir)
