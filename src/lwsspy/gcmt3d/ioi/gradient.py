import os
import numpy as np
from .data import read_data_processed
from .forward import read_synt
from .kernel import read_frechet
from .model import read_model


def write_gradient(g, graddir, it, ls=None):
    if ls is not None:
        fname = f"grad_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"grad_it{it:05d}.npy"
    file = os.path.join(graddir, fname)
    np.save(file, g)


def read_gradient(graddir, it, ls=None):
    if ls is not None:
        fname = f"grad_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"grad_it{it:05d}.npy"
    file = os.path.join(graddir, fname)
    return np.load(file)


def gradient(modldir, graddir, syntdir, datadir, frecdir, it, ls=None):

    # Compute residual
    m = read_model(modldir, it, ls)
    data = read_data_processed(datadir)
    synt = read_synt(syntdir, it, ls)
    resi = synt.flatten() - data.flatten()

    # Gradient
    g = np.zeros(m.size)

    # read each Frechet derivative multiple with the residual and
    for _j in range(m.size):
        dsi_dmj = read_frechet(_j, frecdir, it, ls).flatten()
        g[_j] = 1/dsi_dmj.size * np.sum(dsi_dmj * resi)

    # Write gradient to disk
    write_gradient(g, graddir, it, ls)

    print("      g: ", np.array2string(g, max_line_width=int(1e10)))
