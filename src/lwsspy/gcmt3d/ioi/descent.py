import os
import numpy as np
from .model import read_model
from .gradient import read_gradient
from .hessian import read_hessian
from lwsspy.utils.io import read_yaml_file


def write_descent(dm, descdir, it, ls=None):
    if ls is not None:
        fname = f"desc_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"desc_it{it:05d}.npy"
    file = os.path.join(descdir, fname)
    np.save(file, dm)


def read_descent(descdir, it, ls=None):
    if ls is not None:
        fname = f"desc_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"desc_it{it:05d}.npy"
    file = os.path.join(descdir, fname)
    return np.load(file)


def descent(outdir, damping, it, ls=None):

    # Define the directories
    metadir = os.path.join(outdir, 'meta')
    modldir = os.path.join(outdir, 'modl')
    graddir = os.path.join(outdir, 'grad')
    hessdir = os.path.join(outdir, 'hess')
    descdir = os.path.join(outdir, 'desc')

    # Get damping value
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Read model, gradient, hessian
    m = read_model(modldir, it, ls)
    g = read_gradient(graddir, it, ls)
    H = read_hessian(hessdir, it, ls)

    # Read scaling
    s = np.load(os.path.join(metadir, 'scaling.npy'))

    # Scaling of the cost function
    g *= s
    H = np.diag(s) @ H @ np.diag(s)

    # Get direction
    dm = np.linalg.solve(H + damping * np.trace(H) /
                         m.size * np.diag(np.ones(m.size)), -g)

    # Write direction to file
    write_descent(dm*s, descdir, it, ls)

    print("      d: ", np.array2string(dm, max_line_width=int(1e10)))
