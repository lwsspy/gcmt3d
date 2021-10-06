import os


def check_submission(cmtdir: str, submissiondir: str, verbose: bool = False):
    """Function that checks an event directory on whether an event has already
    bin submitted or have yet to be submitted.

    Parameters
    ----------
    cmtdir : str
        directory with the events
    submissiondir : str
        directory with the submitted events
    verbose : bool, optional
        verbose flag, by default False

    Returns
    -------
    str
        ID of the Event to by submitted next.
    """

    # Check cmts
    cmts = sorted(os.listdir(cmtdir))

    # Check with IDs already submitted
    submittedcmts = os.listdir(submissiondir)

    # Check whether in submission directory
    for event in cmts:
        event
        if event in submittedcmts:
            continue
        else:
            event_to_be_submitted = event
            break

    if "event_to_be_submitted" not in locals():
        if verbose:
            print("No events left to be submitted")

        return "DONE"

    else:
        if verbose:
            print("Next:", event_to_be_submitted)

        return event_to_be_submitted


def bin():

    import sys

    cmtdir = sys.argv[1]
    submitteddir = sys.argv[2]

    print(check_submission(cmtdir, submitteddir))
