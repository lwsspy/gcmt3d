# %%
import os
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.get_data import stage_data
from lwsspy.gcmt3d.ioi.functions.utils import optimdir, wcreate_forward_dirs
from lwsspy.gcmt3d.ioi.functions.forward import update_cmt_synt
from lwsspy.gcmt3d.ioi.functions.kernel import update_cmt_dsdm
from lwsspy.gcmt3d.ioi.functions.processing import process_data, window, process_synt, wprocess_dsdm
from lwsspy.gcmt3d.ioi.functions.model import get_simpars, read_model_names
from lwsspy.gcmt3d.ioi.functions.weighting import compute_weights as compute_weights_func
from lwsspy.gcmt3d.ioi.functions.cost import cost
from lwsspy.gcmt3d.ioi.functions.descent import descent
from lwsspy.gcmt3d.ioi.functions.gradient import gradient
from lwsspy.gcmt3d.ioi.functions.hessian import hessian
from lwsspy.gcmt3d.ioi.functions.linesearch import linesearch as get_optvals, check_optvals
from lwsspy.gcmt3d.ioi.functions.opt import check_done, update_model, update_mcgh
from lwsspy.gcmt3d.ioi.functions.log import update_iter, update_step, reset_step, get_iter
from lwsspy.gcmt3d.ioi.functions.events import check_events_todo


# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):
    node.concurrent = True

    events = check_events_todo(node.inputfile)

    for event in events:
        eventname = CMTSource.from_CMTSOLUTION_file(event).eventname
        out = optimdir(node.inputfile, event, get_dirs_only=True)
        outdir = out[0]
        node.add(cmtinversion, concurrent=False, name=eventname,
                 outdir=outdir, inputfile=node.inputfile,
                 event=event, eventname=eventname,
                 log=os.path.join(outdir, 'logs'))
# -----------------------------------------------------------------------------


# ---------------------------- CMTINVERSION ----------------------------------- 

# Performs inversion for a single event
def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')
    node.add(iteration, concurrent=False)


# Performs iteration
def iteration(node: Node):

    try:
        # Will fail if ITER.txt does not exist
        firstiterflag = get_iter(node.outdir) == 0

    except Exception:
        firstiterflag = True

    if firstiterflag:

        # Create the inversion directory/makesure all things are in place
        node.add_mpi(
            wcreate_forward_dirs, 1, (4, 0), arg=(node.event, node.inputfile),
            name=f"mpi-create-dir-{node.eventname}",
            cwd=node.log)

        # Forward and frechet modeling
        node.add(forward_frechet, concurrent=True)

        # Process the data and the synthetics
        node.add(process_all, concurrent=True, name='processing-all', cwd=node.log)

        # Windowing
        node.add_mpi(window, 1, (10, 0), arg=(node.outdir), cwd=node.log)

        # Weighting
        node.add(compute_weights)

        # Cost, Grad, Hess
        node.add(compute_cgh, concurrent=True)

    # Get descent direction
    node.add(compute_descent)

    # First set of optimization values only computes the initial q and
    # sets alpha to 1
    node.add(compute_optvals)

    node.add(linesearch)

    node.add(iteration_check)


# Performs linesearch
def linesearch(node):
    node.add(search_step)


def search_step(node):
    update_step(node.outdir)
    node.add(compute_new_model)
    node.add(forward_frechet, concurrent=True, outdir=node.outdir)
    node.add(process_synthetics, concurrent=True, name='processing-synthetics', cwd='./logs')
    node.add(compute_cgh, concurrent=True)
    node.add(compute_descent)
    node.add(compute_optvals)
    node.add(search_check)


# ----------------------------- AUXILIARY NODES -------------------------------

# -------------------
# Forward Computation
def forward_frechet(node):
    node.add(forward, concurrent=True)
    node.add(frechet, concurrent=True)


# Forward synthetics
def forward(node):
    # setup
    update_cmt_synt(node.outdir)
    node.add_mpi('bin/xspecfem3D', node.specfem['mpis'], (1, node.specfem['gpus']),
                 cwd=os.path.join(node.outdir, 'simu', 'synt'))


# Frechet derivatives
def frechet(node):
    # Setup
    update_cmt_dsdm(node.outdir)

    # Process the frechet derivatives
    simpars = get_simpars(node.outdir)
    for _i in simpars:
        node.add_mpi('bin/xspecfem3D', node.specfem['mpis'], (1, node.specfem['gpus']),
                     cwd=os.path.join(node.outdir, 'simu', 'dsdm', f'dsdm{_i:05d}'))


# ----------
# Processing
def process_all(node):

    node.add_mpi(process_data, 1, (10, 0), arg=(
        node.outdir), name=node.eventname + '_process_data', cwd=node.log)
    node.add(process_synthetics, concurrent=True)


# Process forward & frechet
def process_synthetics(node):

    # Process the normal synthetics
    node.add_mpi(process_synt, 1, (10, 0),
                 arg=(node.outdir),
                 name=node.eventname + '_process_synt',
                 cwd=node.log)

    # Process the frechet derivatives
    NM = len(read_model_names(node.outdir))
    for _i in range(NM):
        print(node.outdir, 'simpar', _i)
        node.add_mpi(wprocess_dsdm, 1, (10, 0),
                     arg=(node.outdir, _i),
                     name=node.eventname + f'_process_dsdm{_i:05d}',
                     cwd=node.log)

# ------------------
# Updating the model

# update linesearch
def compute_new_model(node):
    update_model(node.outdir)


# Transer to next iteration
def transfer_mcgh(node):
    node.add_mpi(
        update_mcgh, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-transfer-mcgh-{node.eventname}",
        cwd=node.log)


# -------------
# Pre-inversion
def compute_weights(node):
    node.add_mpi(
        compute_weights_func, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-compute-weights-{node.eventname}",
        cwd=node.log)


# --------------------------------
# Cost, Gradient, Hessian, Descent
def compute_cgh(node):
    node.add(compute_cost)
    node.add(compute_gradient)
    node.add(compute_hessian)


# Cost
def compute_cost(node):
    node.add_mpi(
        cost, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-compute-cost-{node.eventname}",
        cwd=node.log)


# Gradient
def compute_gradient(node):
    node.add_mpi(
        gradient, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-compute-grad-{node.eventname}",
        cwd=node.log)


# Hessian
def compute_hessian(node):
    node.add_mpi(
        hessian, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-compute-hess-{node.eventname}",
        cwd=node.log)


# Descent
def compute_descent(node):
    node.add_mpi(
        descent, 1, (4, 0), arg=(node.outdir),
        name=f"mpi-compute-descent-{node.eventname}",
        cwd=node.log)


# ----------
# Linesearch
def compute_optvals(node):
    get_optvals(node.outdir)

# ----------------
# Inversion checks

# Check whether to add another iteration
def iteration_check(node):

    flag = check_optvals(node.outdir, status=False)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        update_iter(node.outdir)
        reset_step(node.outdir)
        if check_done(node.outdir) is False:
            update_iter(node.outdir)
            reset_step(node.outdir)
            node.parent.parent.add(iteration)


def search_check(node):
    # Check linesearch result.
    flag = check_optvals(node.outdir)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        # If linesearch was successful, transfer model
        node.add(transfer_mcgh)

    elif flag == "ADDSTEP":
        # Update step
        node.parent.parent.add(search_step)
