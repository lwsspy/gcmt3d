import imp
import os
from copy import deepcopy
import numpy as np
from .constants import Constants
from .model import read_model, read_model_names, read_perturbation
from .metadata import read_metadata
from lwsspy.seismo.source import CMTSource


def write_frechet(frec, param, frecdir, it, ls=None):
    if ls is not None:
        fname = f"frec{param:05d}_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"frec{param:05d}_it{it:05d}.npy"
    file = os.path.join(frecdir, fname)
    np.save(file, frec)


def read_frechet(param, frecdir, it, ls=None):
    if ls is not None:
        fname = f"frec{param:05d}_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"frec{param:05d}_it{it:05d}.npy"
    file = os.path.join(frecdir, fname)
    return np.load(file)


def update_cmt_dsdm(modldir, metadir, sfredir, it, ls):

    # Read metadata and model
    m = read_model(modldir, it, ls)
    model_names = read_model_names(metadir)

    # Read perturbation
    perturbation = read_perturbation(metadir)

    # Read original CMT solution
    cmt = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmt, _mname, _m)

    # For the perturbations it's slightly more complicated.
    for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

        if _mname not in Constants.nosimpars:

            cmtfiledest = os.path.join(
                sfredir, "dsdm{_i:05d}", "DATA", "CMTSOLUTION")
            if _pert is not None:
                # Perturb source at parameter
                cmt_pert = deepcopy(cmt)

                # If parameter a part of the tensor elements then set the
                # rest of the parameters to 0.
                if _mname in Constants.mt_params:
                    for _tensor_el in Constants.mt_params:
                        if _tensor_el != _mname:
                            setattr(cmt_pert, _tensor_el, 0.0)
                        else:
                            setattr(cmt_pert, _tensor_el, _pert)
                else:

                    # Get the parameter to be perturbed
                    to_be_perturbed = getattr(cmt_pert, _mname)

                    # Perturb the parameter
                    to_be_perturbed += _pert

                    # Set the perturb
                    setattr(cmt_pert, _mname, to_be_perturbed)

                cmt_pert.write_CMTSOLUTION_file()
            else:
                cmt.write_CMTSOLUTION_file(os.path.join(
                    _pardir, "DATA", "CMTSOLUTION"))

        # Write CMTSOLUTION to simulation DATA directory
    cmt.write_CMTSOLUTION_file(
        os.path.join(ssyndir, 'DATA', 'CMTSOLUTION'))


def frechet(param: int, modldir, metadir, frecdir, it, ls):

    # Read metadata and model
    m = read_model(modldir, it, ls)
    X = read_metadata(metadir)

    # Forward modeling
    frechet = dgdm(m, X, param)

    # Write Frechet derivative
    write_frechet(frechet, param, frecdir, it, ls)
