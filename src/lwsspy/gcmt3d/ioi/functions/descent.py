import os
import numpy as np
from .model import read_model
from .gradient import read_gradient
from .hessian import read_hessian
from lwsspy.utils.io import read_yaml_file
from .log import get_iter, get_step


def write_descent(dm, outdir, it, ls=None):

    # Get graddir
    descdir = os.path.join(outdir, 'desc')

    # Get filename
    if ls is not None:
        fname = f"dm_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"dm_it{it:05d}.npy"

    # Full filename
    file = os.path.join(descdir, fname)

    # Save
    np.save(file, dm)


def read_descent(outdir, it, ls=None):

    # Get graddir
    descdir = os.path.join(outdir, 'desc')

    if ls is not None:
        fname = f"dm_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"dm_it{it:05d}.npy"

    file = os.path.join(descdir, fname)

    return np.load(file)


def descent(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Define the directories
    metadir = os.path.join(outdir, 'meta')

    # Get damping value
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get damping value
    damping = inputparams['optimization']['damping']

    # Read model, gradient, hessian
    m = read_model(outdir, it, ls)
    g = read_gradient(outdir, it, ls)
    H = read_hessian(outdir, it, ls)

    # Read scaling
    s = np.load(os.path.join(metadir, 'scaling.npy'))

    # Scaling of the cost function
    g *= s
    H = np.diag(s) @ H @ np.diag(s)

    # Get direction
    dm = np.linalg.solve(H + damping * np.trace(H) /
                         m.size * np.diag(np.ones(m.size)), -g)

    # Write direction to file
    write_descent(dm*s, outdir, it, 0)

    print("      d: ", np.array2string(dm, max_line_width=int(1e10)))
