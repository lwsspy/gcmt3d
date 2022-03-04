import doctest
import os
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.cmt_catalog import CMTCatalog
from lwsspy.utils.io import read_yaml_file, write_yaml_file
from .utils import createdir
from .log import read_status

def write_event_status(dir, eventname, message):
    with open(os.path.join(dir, eventname), 'w') as f:
        f.write(message)

def read_event_status(dir, eventname):
    with open(os.path.join(dir, eventname), 'r') as f:
        message = f.read()
    return message


def create_event_status_dir(eventdir, inputfile):

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)
    downdir = os.path.join(neweventdir, "DOWNLOADED")
    initdir = os.path.join(neweventdir, "EVENTS_INIT")
    out_dir = os.path.join(neweventdir, "EVENTS_FINAL")
    statdir = os.path.join(neweventdir, "STATUS")

    # Create dirs if they don't exist
    createdir(downdir)
    createdir(initdir)
    createdir(out_dir)
    createdir(statdir)

    # Write input file to eventdir
    write_yaml_file(inputparams, os.path.join(neweventdir, 'input.yml'))
    
    # Eventfiles
    eventfilelist = os.listdir(eventdir)

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Write CMT's to directory
    cat.cmts2dir(initdir)


def add_events(eventdir, inputfile):
    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Directory with cmt solutions on file
    initdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Eventfiles
    eventfilelist = os.listdir(eventdir)

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    for _cmt in cat:

        # Eventname
        cmtname = _cmt.eventname

        # Location in eventstatusdir
        dst = os.path.join(initdir, cmtname)

        # Skip if file already in the events directory
        if os.path.exists(dst):
            continue
        else:
            # Write CMT's to directory
            cat.cmts2file(dst)


def check_events(inputfile):
    """
    Can only be run after the create_event_status_dir. The inputfile
    should be the one in the Event status directory.

    This function is integral to the workflow since it marks which events have 
    to be run and which dont.
    """

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get database directory
    database = inputparams["database"]

    # Get location of the observed data
    datadatabase = inputparams["datadatabase"]

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Path to event directory
    eventdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Status dir
    statdir = os.path.join(neweventdir, "STATUS")

    # down dir
    downdir = os.path.join(neweventdir, "DOWNLOADED")

    # Get list of eventfile
    eventfilelist = os.listdir(eventdir)

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Copy event to the init events if it doesn't exist yet
    # Check whether downloaded and if yes, put 1 in to file in
    # DOWNLOADED 
    for _cmt in cat:

        # Eventname
        cmtname = _cmt.eventname

        # CMT database directory
        db_cmt = os.path.join(database, cmtname)

        # CMT data directory
        data_cmt = os.path.join(datadatabase, cmtname)

        try:
            # Check download status    
            downstat = read_status(data_cmt)

            if downstat == 'FAILED':
                write_event_status(downdir, cmtname, 'FAIL')
                write_event_status(statdir, cmtname, 'CANT')
                continue
            elif (downstat == 'DOWNLOADING'):
                write_event_status(downdir, cmtname, 'UNFINISHED')
                write_event_status(statdir, cmtname, 'CANT')
                continue
            elif (downstat == 'CREATED'):
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
                write_event_status(statdir, cmtname, 'CANT')
                continue
            elif (downstat == 'DOWNLOADED'):
                write_event_status(downdir, cmtname, 'TRUE')
            else:
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')    
                write_event_status(statdir, cmtname, 'CANT')
                continue

        # Not created yet
        except Exception:
            write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
            write_event_status(statdir, cmtname, 'CANT')
            continue

        # Inversion status
        try:
            inv_stat = read_status(db_cmt)

            if "FINISHED" in inv_stat:
                write_event_status(statdir, cmtname, 'DONE')
            elif "FAIL" in inv_stat:
                write_event_status(statdir, cmtname, 'FAIL')
            elif ("SUCCESS" in inv_stat) or ("ADDSTEP" in inv_stat):
                write_event_status(statdir, cmtname, 'RUNNING')
            else:
                write_event_status(statdir, cmtname, 'TODO')
                
        except Exception:
            write_event_status(statdir, cmtname, 'TODO')


def check_events_todo(inputfile):
    """Can only be run after the ``create_event_status_dir``. The inputfile
    should be the one in the Event status directory."""

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Path to event directory
    eventdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Status dir
    statdir = os.path.join(neweventdir, "STATUS")

    # Get list of eventfile
    eventfilelist = os.listdir(eventdir)

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Copy event to the init events if it doesn't exist yet
    # Check whether downloaded and if yes, put 1 in to file in
    # DOWNLOADED
    TODO = []
    for _i, _cmt in enumerate(cat):

        # Eventname
        cmtname = _cmt.eventname

        # Inversion status
        status = read_event_status(statdir, cmtname)

        # If todo append cmtname
        if status == "TODO":
            TODO.append(eventfilelist[_i])

    return TODO


                
        
        






