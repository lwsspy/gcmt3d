

import os
import glob
from .. import seismo as lseis
from .. import utils as lutil
from obspy import UTCDateTime


def write_filtered_catalog(
        catalog: str, catalog_params_file: str, outdir: str,
        verbose: bool = True):
    """Using a base catalog that is to be filtered and a filter file. This
    writes cmtsolutions of the filtered catalog to a directory.

    Parameters
    ----------
    catalog : str
        filename to the original :class:``lwsspy.seismo.cmt_catalog.CMTCatalog``
    catalog_params_file : str
        filename of a yaml filter file with format looking like

        .. code::

            mindict:
                moment_magnitude: 5.5
                origin_time: 2000-01-01T00:00:00

            maxdict:
                moment_magnitude: 8.0
                origin_time: 2020-01-01T00:00:00

        These parameters follow the naming convention of
        :class:``lwsspy.seismo.source.CMTSource``
    outdir : str
        Directory to write the filtered CMTSolutions to
    verbose : bool, optional
        verbose flag, by default True
    """

    filterdicts = lutil.read_yaml_file(catalog_params_file)

    if verbose:
        print(filterdicts)

    if filterdicts is None:
        filterdicts = dict()

    # Check if mindict in the yaml file
    if "mindict" in filterdicts:

        # Assign the dict
        mindict = filterdicts["mindict"]

        # Get timestamp to UTC
        if "origin_time" in mindict:
            mindict["origin_time"] = UTCDateTime(mindict["origin_time"])

    else:
        mindict = None

    # Check if mindict in the yaml file
    if "maxdict" in filterdicts:

        # Assign the dict
        maxdict = filterdicts["maxdict"]

        # Get timestamp to UTC
        if "origin_time" in mindict:
            maxdict["origin_time"] = UTCDateTime(maxdict["origin_time"])

    else:
        maxdict = None

    # Check
    if verbose:
        print("Max")
        print(maxdict)

    if verbose:
        print("Min")
        print(mindict)

    # Load catalog
    cat = lseis.CMTCatalog.load(catalog)

    # Load
    if maxdict is None and mindict is None:
        filtered_cat = cat
    else:
        filtered_cat = cat.filter(maxdict=maxdict, mindict=mindict)

    if verbose:
        print("Original:", len(cat))
        print("Filtered:", len(filtered_cat))

    # Sort the catalog
    filtered_cat.sort(key='eventname')

    # Write
    filtered_cat.cmts2dir(outdir=outdir)


def bin():

    import sys

    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
        string = """

Usage:
    gcmt3d-write-filtered-cat <original-catalog.pkl> <filter-parameters.yml> <outputdir/>

Example:
    gcmt3d-write-filtered-cat gcmtcatalog.pkl catalog_params.yml cmtsolutions

        """
        print(string)

        sys.exit()

    else:
        catalog = sys.argv[1]
        catalog_params_file = sys.argv[2]
        outdir = sys.argv[3]

    write_filtered_catalog(catalog, catalog_params_file, outdir)
