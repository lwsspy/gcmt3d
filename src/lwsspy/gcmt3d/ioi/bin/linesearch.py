"""

Exectuable that checks and writes linesearch parameters for on linesearch
iteration.

Usage:
    ioi-linesearch <optdir> <descdir> <graddir> <costdir> <it> <ls>

where:
    optdir   - directory containing the optimization parameters
    it       - iteration number
    ls       - linesearch number

"""

from sys import exit, argv
from ..functions.linesearch import linesearch


def bin():

    if len(argv) != 6+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    optdir, descdir, graddir, costdir, it, ls = argv[1:7]

    # Clearlog
    linesearch(optdir, descdir, graddir, costdir, it, ls)
