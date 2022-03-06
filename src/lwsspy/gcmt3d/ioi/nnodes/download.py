# %%
from pprint import pprint
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.get_data import get_data
from lwsspy.gcmt3d.ioi.functions.utils import downloaddir
from lwsspy.gcmt3d.ioi.functions.events import check_events_todownload


# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODOWNLOAD event check
def main(node: Node):

    node.concurrent = True

    for event in check_events_todownload(node.inputfile):
        # event = read_events(eventdir)
        eventname = CMTSource.from_CMTSOLUTION_file(event).eventname
        out = downloaddir(node.inputfile, event, get_dirs_only=True)
        outdir = out[0]
        node.add(download, concurrent=True, name=eventname + "-Download",
                 outdir=outdir, event=event, eventname=eventname,
                 log='./logs/' + eventname)
# -----------------------------------------------------------------------------


# ---------------------------- DATA DOWNLOAD ---------------------------------- 
def download(node: Node):

    # Create base dir
    _ = downloaddir(node.inputfile, node.event)

    # Download data
    node.add_mpi(
        get_data, 1, (4, 0), arg=(node.outdir), cwd=node.log,
        name=f"mpi-get-data-{node.eventname}")


