from typing import Iterable, Sequence
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.utils import downloaddir
from lwsspy.gcmt3d.ioi.functions.get_data_mpi import get_data_mpi
from lwsspy.gcmt3d.ioi.functions.events import check_events_todownload


def split(sequence: Sequence, count: int):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.

    sequence = sequence to be split    
    count = number of chunks to split the sequence into

    return list of count chunks where the number of elements per chunk depends
    on the length of the sequence
    """
    return [sequence[_i::count] for _i in range(count)]


def chunkfunc(sequence: Sequence, n: int):
    """
    Converse to split, this function preserves the order of the events. But,
    the last chunk has potentially only a single element. For the case, of 
    I/O this is not so bad, but if you are looking for performance, `split()`
    above is the better option.

    sequence = sequence to be split    
    n = number of elements per chunk

    return list of n-element chunks where the number of chunks depends on the 
    length of the sequence
    """
    n = max(1, n)
    return [sequence[i:i+n] for i in range(0, len(sequence), n)]


# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODOWNLOAD event check
def main(node: Node):

    node.concurrent = False

    # Maximum download flag
    maxflag = True if node.max_downloads != 0 else False

    eventfiles = check_events_todownload(node.inputfile)

    if maxflag:
        eventfiles = eventfiles[:node.max_downloads]

    # Node download MPI or not
    if node.download_mpi == 0:

        # Find number of chunks by
        # Nchunks = round(len(eventfiles)/int(node.events_per_chunk))
        eventfile_chunks = chunkfunc(eventfiles, node.events_per_chunk)

        for chunk in eventfile_chunks:
            node.add(download_chunks, concurrent=True, eventfiles=chunk)


    else:

        node.add_mpi(
            get_data_mpi, node.download_mpi, (1, 0),
            arg=(eventfiles, node.inputfile),
            name=f"Download-{len(eventfiles)}-events-on-{node.download_mpi}-cores")


def download_chunks(node: Node):

    for event in node.eventfiles:

            # event = read_events(eventdir)
            eventname = CMTSource.from_CMTSOLUTION_file(event).eventname
            out = downloaddir(node.inputfile, event, get_dirs_only=True)
            outdir = out[0]

            node.add(download, concurrent=True, name=eventname + "-Download",
                     outdir=outdir, event=event, eventname=eventname)

async def download(node: Node):

    # Create base dir
    await node.call_async(f'gcmt3d-get-data {node.event} {node.inputfile}')

