"""

Exectuable that checks the status of the inversion after linesearch.

Usage:
    ioi-check-status <statusdir>

where:
   statusdir  -  directory 'STATUS.txt' is written to

"""

from sys import exit, argv
from ..opt import check_status


def bin():

    if len(argv) != 1+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    statusdir = argv[1]

    # Check Status
    print(check_status(statusdir))
