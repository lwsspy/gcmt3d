"""

Exectuable that checks teh optimization parameters on whether further iteration
 or linesearch is necessary.

Usage:
    opt-check-optvals <optdir> <statdir> <costdir> <it> <ls> <nls_maxs>
                
where:
    optdir   - directory containing the optimization parameters
    statdir  - directory containing the 'STATUS.txt'
    costdir  - directory containing the costs
    it       - iteration number
    ls       - linesearch number
    nls_max  - max number of linesearches

"""

from sys import exit, argv
from ..linesearch import check_optvals


def bin():

    if len(argv) != 6+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    optdir, statdir, costdir, it, ls, nls_max = argv[1:7]

    # Clearlog
    check_optvals(optdir, statdir, costdir, it, ls, nls_max)
