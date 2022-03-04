import os
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.cmt_catalog import CMTCatalog
from lwsspy.utils.io import read_yaml_file
from .utils import createdir

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
    statdir = os.path.join(neweventdir, "STATUS")
    initdir = os.path.join(neweventdir, "INITEVENTS")
    out_dir = os.path.join(neweventdir, "OUT_EVENTS")

    # Create dirs if they don't exist
    createdir(statdir)
    createdir(initdir)
    createdir(out_dir)
    
    # 
    






