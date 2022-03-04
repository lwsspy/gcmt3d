# %%
import os
from pprint import pprint
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.get_data import get_data
from lwsspy.gcmt3d.ioi.functions.utils import optimdir, downloaddir, read_events
from lwsspy.gcmt3d.ioi.functions.constants import Constants

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):

    inputfile = node.inputfile

    node.concurrent = True

    print("\n node attribs \n")
    pprint(node.__dict__)

    print("\n parent attribs \n")
    try:
        pprint(node.parent.__dict__)
    except Exception as e:
        print(e)

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

