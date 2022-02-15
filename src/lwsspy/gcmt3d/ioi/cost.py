import os
import numpy as np
from .data import read_data_processed
from .forward import read_synt


def write_cost(c, costdir, it, ls=None):
    if ls is not None:
        fname = f"cost_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"cost_it{it:05d}.npy"
    file = os.path.join(costdir, fname)
    np.save(file, c)


def read_cost(costdir, it, ls=None):
    if ls is not None:
        fname = f"cost_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"cost_it{it:05d}.npy"
    file = os.path.join(costdir, fname)
    return np.load(file)


def cost(datadir, syntdir, costdir, it, ls=None):

    # Compute residual
    data = read_data_processed(datadir)
    synt = read_synt(syntdir, it, ls)
    resi = synt.flatten() - data.flatten()

    c = 0.5/resi.size * np.sum((resi)**2)

    write_cost(c, costdir, it, ls)

    print("      c:", np.array2string(c, max_line_width=int(1e10)))
