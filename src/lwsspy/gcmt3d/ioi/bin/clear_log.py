"""

 Executable that clears the log in a given log directory.

Usage:
    ioi-clear-log optdir

where:
    logdir   - directory containing the log(s)
    
"""

from sys import exit, argv
from ..functions.log import clear_log


def bin_clear_log():

    if len(argv) != 1+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get log dir from command line arguments
    optdirdir = argv[1]

    # Clearlog
    clear_log(optdir)
