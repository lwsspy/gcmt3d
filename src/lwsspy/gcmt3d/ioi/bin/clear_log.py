
from sys import exit, argv
from ..opt import check_done, check_status, update_model, update_mcgh
from ..log import clear_log

"""

 Executable that clears the log in a given log directory.

Usage:
    opt-clear-log <logdir>

where:
    logdir   - directory containing the log(s)
    
"""


def bin_clear_log():

    if len(argv) != 1+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get log dir from command line arguments
    logdir = argv[1]

    # Clearlog
    clear_log(logdir)
