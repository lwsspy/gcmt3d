"""

Exectuable that checks whether an iteration should be added.

Usage:
    ioi-check-done <optdir> <it> <ls>

where:
    optdir   - directory containing the optimization parameters
    it       - iteration number
    ls       - linesearch number

"""

from sys import exit, argv
from ..opt import check_done


def bin():

    if len(argv) != 1+3:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    optdir, it, ls = argv[1:4]

    # Clearlog
    check_done(optdir, it, ls)
