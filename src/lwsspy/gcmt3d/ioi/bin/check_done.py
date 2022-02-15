"""

Exectuable that checks and writes linesearch parameters for on linesearch
iteration.

Usage:
    opt-linesearch <optdir> <descdir> <graddir> <costdir> <it> <ls>
optdir, modldir, costdir, descdir, statdir, it, ls,
        stopping_criterion=0.01,
        stopping_criterion_model=0.01,
        stopping_criterion_cost_change=0.001
where:
    optdir   - directory containing the optimization parameters
    descdir  - directory containing the descent directions
    graddir  - directory containing the gradients
    costdir  - directory containing the costs
    it       - iteration number
    ls       - linesearch number

"""

from sys import exit, argv
from ..opt import check_done


def bin():

    if len(argv) != 6+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    optdir, descdir, graddir, costdir, it, ls = argv[1:7]

    # Clearlog
    check_done(optdir, modldir, costdir, descdir, statdir, it, ls)
