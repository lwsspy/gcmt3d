from curses import meta
import imp
import os
from lwsspy.utils.io import read_yaml_file
import numpy as np
from .model import read_model, read_model_names
from .metadata import read_metadata
from .gaussian2d import g
from lwsspy.seismo.source import CMTSource


def write_synt(synt, syntdir, it, ls=None):
    if ls is not None:
        fname = f"synt_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"synt_it{it:05d}.npy"
    file = os.path.join(syntdir, fname)
    np.save(file, synt)


def read_synt(syntdir, it, ls=None):
    if ls is not None:
        fname = f"synt_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"synt_it{it:05d}.npy"
    file = os.path.join(syntdir, fname)
    return np.load(file)


def update_cmt_synt(modldir, metadir, ssyndir, it, ls):

    # Read metadata and model
    m = read_model(modldir, it, ls)
    model_names = read_model_names(metadir)

    # Read original CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt')
    )

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmtsource, _mname, _m)

    # Write CMTSOLUTION to simulation DATA directory
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(ssyndir, 'DATA', 'CMTSOLUTION'))


# def forward(outdir, ssyndir, it, ls=None):

#     # Read input file
#     inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

#     # Launchmethod
#     flag = "-t00:05:00"
#     launch_method = inputparams['launch_method'] + " {flag}"

    #
