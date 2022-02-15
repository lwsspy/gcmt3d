import os
from nbformat import read
import numpy as np
from lwsspy.utils.io import read_yaml_file, write_yaml_file
from copy import copy, deepcopy

from py import process
from .constants import Constants
from .data import write_data, write_data_processed
from .model import write_model
from .metadata import write_metadata
from .gaussian2d import g
from .utils import adapt_processdict
from lwsspy.seismo.source import CMTSource

# %% Get Model CMT


def get_data(cmtfile, outdir, datadir, modldir, metadir, scaldir, inputparamfile):

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Download params
    download_dict = dict(
        network=",".join(['CU', 'G', 'GE', 'IC', 'II', 'IU', 'MN']),
        channel_priorities=["LH*", "BH*"],
    )

    # Read parameterfile
    inputparams = read_yaml_file(inputparamfile)

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Get initial processing directory
    if inputparams["processparams"] is None:
        processdict = Constants.processdict
    else:
        processdict = read_yaml_file(inputparams['processparams'])

    # Adapting the processing dictionary
    processdict = adapt_processdict(cmtsource, processdict, duration)

    # Writing the new processing file to the directory
    write_yaml_file(processdict, os.path.join(outdir, 'process.yml'))

    return None

    # actual m: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
    mdictsol = dict(
        a=4,
        x0=115,
        y0=90,
        sigma_x=25,
        sigma_y=35,
        theta=0.0,
        c=2
    )
    m_sol = np.array([val for val in mdictsol.values()])

    # Dictionary for the scaling
    scaling_dict = dict(
        a=1,
        x0=25,
        y0=35,
        sigma_x=25,
        sigma_y=35,
        theta=0.1,
        c=0.1
    )

    scaling = np.array([val for val in scaling_dict.values()])

    # Initial guess m0: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
    mdict_init = dict(
        a=1,
        x0=100,
        y0=100,
        sigma_x=20,
        sigma_y=40,
        theta=0.1,
        c=0.1
    )
    m_init = np.array([val for val in mdict_init.values()])

    # Create tracked model vectors
    m = copy(m_init)
    mdict = deepcopy(mdict_init)
    mnames = [key for key in mdict.keys()]

    # %%

    # Write model to disk
    mdict = {key: m[_i] for _i, key in enumerate(mdict_init)}

    # %%
    # write model
    it = 0
    ls = 0

    # %%
    # Create some data
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    X = np.meshgrid(x, y)

    # %%

    # Create data
    data = g(m_sol, X)
    data_processed = data + \
        (np.random.normal(loc=0.0, scale=0.5, size=data.shape))

    # Set linesearch and iteration number to 0
    it, ls = 0, 0

    # Write model
    write_model(m, modldir, it, ls)

    # Write scaling of the model
    np.save(os.path.join(scaldir, 'scaling.npy'), scaling)

    # Metadatadir
    x, y = X
    write_metadata(x, y, metadir)

    # Write data
    write_data(data, datadir)
    write_data_processed(data_processed, datadir)
