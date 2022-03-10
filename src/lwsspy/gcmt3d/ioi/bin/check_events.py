"""

Usage: 

    gcmt3d-check-events <path/to/input.yml>

This script calls a python function that checks the events statuses and updates
the statuses in the event status dir.

Should be run after gcmt3d-create-event-dir

--- 

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import argv, exit
from ..functions.events import check_events


def bin():

    # Get args or print usage statement
    if (len(argv) == 1) or (len(argv) > 4):
        print(__doc__)
        exit()
    elif (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    elif len(argv) == 2:
        inputfile = argv[1]
        # Check event statuses
        check_events(inputfile)
    elif len(argv) == 3:
        inputfile, resetflag = argv[1:]
        check_events(inputfile, resetopt=resetflag)
    else:
        print(__doc__)
        exit()
