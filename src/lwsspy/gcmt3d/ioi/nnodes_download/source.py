# %%
import os
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.get_data import get_data
from lwsspy.gcmt3d.ioi.functions.utils import optimdir, downloaddir, read_events

# %%
eventdir = "/home/lsawade/events"
inputfile = "/home/lsawade/lwsspy/lwsspy.gcmt3d/src/lwsspy/gcmt3d/ioi/input.yml"
processfile = "/home/lsawade/lwsspy/lwsspy.gcmt3d/src/lwsspy/gcmt3d/ioi/process.yml"

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):
    node.concurrent = True

    for event in read_events(eventdir):
        # event = read_events(eventdir)
        eventname = CMTSource.from_CMTSOLUTION_file(event).eventname
        out = downloaddir(inputfile, event, get_dirs_only=True)
        outdir = out[0]
        node.add(download, concurrent=True,
                 outdir=outdir, inputfile=inputfile,
                 event=event, eventname=eventname,
                 log='./logs/' + eventname)
# -----------------------------------------------------------------------------



# ---------------------------- DATA DOWNLOAD ---------------------------------- 
def download(node: Node):

    # Create base dir
    _ = downloaddir(node.inputfile, node.event)

    # Download data
    get_data(node.outdir)

