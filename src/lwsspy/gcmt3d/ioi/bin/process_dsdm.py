"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import argv, exit
from ..functions.processing import process_dsdm


def bin():
    """

    Usage:

        gcmt3d-process-dsdm eventdir param

    This script calls a python function that process the frechet derivatives and
    saves them in the frec/ folder

    """

    # Get args or print usage statement 
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        outdir, nm = argv[1:]

    # Run the initializer
    process_dsdm(outdir, nm)
