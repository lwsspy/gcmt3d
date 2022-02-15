import os
import numpy as np
from .model import read_model, write_model
from .cost import read_cost, write_cost
from .descent import read_descent
from .gradient import read_gradient, write_gradient
from .hessian import read_hessian, write_hessian
from .linesearch import read_optvals
from .log import write_status


def read_optparams(paramdir):
    pass


def update_model(modldir, descdir, optdir, it, ls):

    # Read model, descent direction, and optvals (alpha)
    m = read_model(modldir, it, ls)
    dm = read_descent(descdir, it, ls)
    _, _, _, alpha, _, _, _ = read_optvals(optdir, it, ls)

    # Compute new model
    m_new = m + alpha * dm

    # Write new model
    write_model(m_new, modldir, it, ls + 1)

    print("      m: ", np.array2string(m_new, max_line_width=int(1e10)))


def update_mcgh(modldir, costdir, graddir, hessdir, it, ls):

    # Read all relevant data
    m = read_model(modldir, it, ls)
    c = read_cost(costdir, it, ls)
    g = read_gradient(graddir, it, ls)
    h = read_hessian(hessdir, it, ls)

    # Write for the first iteration and 0 ls
    write_model(m, modldir, it + 1, 0)
    write_cost(c, costdir, it + 1, 0)
    write_gradient(g, graddir, it + 1, 0)
    write_hessian(h, hessdir, it + 1, 0)


def check_status(statdir):
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "r") as f:
        message = f.read()

    print("    STATUS:", message)

    if "FAIL" in message:
        return False
    else:
        return True


def check_done(
        optdir, modldir, costdir, descdir, statdir, it, ls,
        stopping_criterion=0.01,
        stopping_criterion_model=0.01,
        stopping_criterion_cost_change=0.001):

    # Read cost
    cost_init = read_cost(costdir, 0, 0)
    cost_old = read_cost(costdir, it, 0)
    cost = read_cost(costdir, it+1, 0)

    # Read necessary vals
    _, _, _, alpha, _, _, _ = read_optvals(optdir, it, ls)
    # descent = read_descent(descdir, it, ls)
    # descent_prev = read_descent(descdir, it, ls-1)
    model = read_model(modldir, it, ls)
    model_prev = read_model(modldir, it, ls-1)
    # model_init = read_model(modldir, 0, 0)
    # scaling = np.load(statdir, 'scaling')

    STATUS = False

    if (np.abs(cost - cost_old)/cost_init < stopping_criterion_cost_change):
        message = "FINISHED: Cost function not decreasing enough to justify iteration."
        write_status(statdir, message)
        STATUS = True
    elif (cost/cost_init < stopping_criterion):
        message = "FINISHED: Optimization algorithm has converged."
        write_status(statdir, message)
        STATUS = True
    elif np.sum(model - model_prev)**2/np.sum(model_prev**2) \
            < stopping_criterion_model:
        message = "FINISHED: Model is not updating enough anymore."
        write_status(statdir, message)
        STATUS = True

    return STATUS
