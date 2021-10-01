"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# External
import os
import sys
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from obspy import read, read_events
from obspy.imaging.beachball import beach

# Internal
from lwsspy import plot_station_xml
from lwsspy import process_wrapper
from lwsspy import read_inventory
from lwsspy import CMTSource
from lwsspy import window_on_stream
from lwsspy import add_tapers
from lwsspy import stream_cost_win
from lwsspy import stream_grad_frechet_win
from lwsspy import stream_grad_and_hess_win
from lwsspy import stream_multiply
from lwsspy import createsimdir
from lwsspy import read_parfile
from lwsspy import stationxml2STATIONS
from lwsspy import write_parfile
from lwsspy import Optimization
from lwsspy import plot_optimization
from lwsspy import plot_model_history
from lwsspy import updaterc
from lwsspy import run_cmds_parallel
from lwsspy import read_yaml_file
from lwsspy import print_action, print_bar, print_section
updaterc()


# Main parameters
scriptdir = os.path.dirname(os.path.abspath(__file__))
station_xml = '/home/lsawade/lwsspy/invdir/station2_filtered.xml'
SPECFEM = "/scratch/gpfs/lsawade/MagicScripts/specfem3d_globe"
specfem_dict = {
    "bin": "link",
    "DATA": {
        "Par_file": "file",
    },
    "DATABASES_MPI": "link",
    "OUTPUT_FILES": "dir"
}

# %% Create inversion directory and simulation directories
invdir = '/home/lsawade/lwsspy/invdir'
if os.path.exists(invdir) is False:
    os.mkdir(invdir)

datasimdir = os.path.join(invdir, 'Data')
syntsimdir = os.path.join(invdir, 'Synt')
dsynsimdir = os.path.join(invdir, 'Dsyn')

datastations = os.path.join(datasimdir, 'DATA', 'STATIONS')
syntstations = os.path.join(syntsimdir, 'DATA', 'STATIONS')
dsynstations = os.path.join(dsynsimdir, 'DATA', 'STATIONS')
data_parfile = os.path.join(datasimdir, "DATA", "Par_file")
synt_parfile = os.path.join(syntsimdir, "DATA", "Par_file")
dsyn_parfile = os.path.join(dsynsimdir, "DATA", "Par_file")
data_cmt = os.path.join(datasimdir, "DATA", "CMTSOLUTION")
synt_cmt = os.path.join(syntsimdir, "DATA", "CMTSOLUTION")
dsyn_cmt = os.path.join(dsynsimdir, "DATA", "CMTSOLUTION")
compute_synt = True
if compute_synt:

    # Create simulation dirs
    createsimdir(SPECFEM, syntsimdir, specfem_dict=specfem_dict)
    createsimdir(SPECFEM, dsynsimdir, specfem_dict=specfem_dict)

    # Set Station file
    stationxml2STATIONS(station_xml, syntstations)
    stationxml2STATIONS(station_xml, dsynstations)

    # Set data parameters
    synt_pars = read_parfile(synt_parfile)
    dsyn_pars = read_parfile(dsyn_parfile)
    synt_pars["USE_SOURCE_DERIVATIVE"] = False
    dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
    dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1  # For depth
    write_parfile(synt_pars, synt_parfile)
    write_parfile(dsyn_pars, dsyn_parfile)

compute_data = False
if compute_data:
    # Once data is created it can be left..
    createsimdir(SPECFEM, datasimdir, specfem_dict=specfem_dict)

    # Set Station file
    stationxml2STATIONS(station_xml, datastations)

    # Set data parameters
    data_pars = read_parfile(data_parfile)
    data_pars["USE_SOURCE_DERIVATIVE"] = False
    write_parfile(data_pars, data_parfile)

# %% Get Model CMT
shallow = False
if shallow:
    eventype = "Shallow"
    cmtfile = "CMTSOLUTION_Italy_shallow"
else:
    eventype = "Deep"
    cmtfile = "CMTSOLUTION"

CMTSOLUTION = os.path.join(invdir, cmtfile)
cmt_goal = CMTSource.from_CMTSOLUTION_file(CMTSOLUTION)
cmt_init = deepcopy(cmt_goal)
xml_event = read_events(CMTSOLUTION)[0]

# Add 10km to initial model
cmt_init.depth_in_m += 10000.0
cmt_goal.write_CMTSOLUTION_file(data_cmt)


# %%
def generate_data(specfemdir):
    """Launches specfem for a forward simulation with the target
    CMTSOLUTION model.
    """

    cmd_list = [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    # This hopefully gets around the change of directory thing
    cwdlist = [specfemdir]
    run_cmds_parallel(cmd_list, cwdlist=cwdlist)


# Running forward simulation
print_bar("Starting the inversion")

if compute_data:
    print_action("Generating the data")
    generate_data(datasimdir)

# sys.exit()

# Loading Station Data
inv = read_inventory(station_xml)

# Loading Seismic Data
rawdata = read(os.path.join(datasimdir, "OUTPUT_FILES", "*.sac"))

# Loading Process Parameters
processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))

# Processing observed data
data = process_wrapper(rawdata, cmt_init, processparams,
                       inv=inv, observed=False)

# Checking how many Traces are left
print(f"Data # of traces: {len(data)}")


# %% cost and gradient computation
iterwindow = 0

# Define function that iterates over depth


def compute_cost_and_gradient(model):
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
    cmt.write_CMTSOLUTION_file(synt_cmt)
    cmt.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    if compute_synt:
        run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing Synthetic")
    # with nostdout():
    synt = process_wrapper(synt, cmt_init, processparams,
                           inv=inv, observed=False)

    print_action("Processing Fréchet")
    # with nostdout():
    dsyn = process_wrapper(dsyn, cmt_init, processparams,
                           inv=inv, observed=False)

    # After the first forward modeling window the data
    if iterwindow == 0:
        print_action("Windowing")
        window_config = read_yaml_file(
            os.path.join(scriptdir, "window.body.yml"))
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad = np.array([stream_grad_frechet_win(data, synt, dsyn)])

    return cost, grad


def compute_cost_and_gradient_hessian(model):
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
    cmt.write_CMTSOLUTION_file(synt_cmt)
    cmt.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    if compute_synt:
        run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing Synthetic")
    # with nostdout():
    synt = process_wrapper(synt, cmt_init, processparams,
                           inv=inv, observed=False)

    print_action("Processing Fréchet")
    # with nostdout():
    dsyn = process_wrapper(dsyn, cmt_init, processparams,
                           inv=inv, observed=False)

    # After the first forward modeling window the data
    if iterwindow == 0:
        print_action("Windowing")
        window_config = read_yaml_file(
            os.path.join(scriptdir, "window.body.yml"))
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad, hess = stream_grad_and_hess_win(data, synt, [dsyn])

    return cost, np.array([grad]), np.array([[hess]])


def compute_cost_and_gradient2(model):
    """Computes the Gradients and hessian with respect to both depth and
    origin time
    """
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
    cmt.cmt_time = cmt.cmt_time + model[1]
    cmt.write_CMTSOLUTION_file(synt_cmt)
    cmt.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    if compute_synt:
        run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing Synthetic")
    # with nostdout():
    synt = process_wrapper(synt, cmt_init, processparams,
                           inv=inv, observed=False)

    print_action("Processing Fréchet")
    # with nostdout():
    dz = process_wrapper(dsyn, cmt_init, processparams,
                         inv=inv, observed=False)
    # Compute the derivative with respect to origin time
    # -> negative time derivative
    ddt = deepcopy(synt)
    ddt.differentiate()
    stream_multiply(ddt, -1.0)

    # After the first forward modeling window the data
    if iterwindow == 0:
        print_action("Windowing")
        window_config = read_yaml_file(
            os.path.join(scriptdir, "window.body.yml"))
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad, _ = stream_grad_and_hess_win(data, synt, [dz, ddt])

    return cost, grad


def compute_cost_and_gradient_hessian2(model):
    """Computes the Gradients and hessian with respect to both depth and
    origin time
    """
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
    cmt.cmt_time = cmt.cmt_time + model[1]
    cmt.write_CMTSOLUTION_file(synt_cmt)
    cmt.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    if compute_synt:
        run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing Synthetic")
    # with nostdout():
    synt = process_wrapper(synt, cmt_init, processparams,
                           inv=inv, observed=False)

    print_action("Processing Fréchet")
    # with nostdout():
    dz = process_wrapper(dsyn, cmt_init, processparams,
                         inv=inv, observed=False)
    # Compute the derivative with respect to origin time
    # -> negative time derivative
    ddt = deepcopy(synt)
    ddt.differentiate()
    # stream_multiply(ddt, -1.0)

    # After the first forward modeling window the data
    if iterwindow == 0:
        print_action("Windowing")
        window_config = read_yaml_file(
            os.path.join(scriptdir, "window.body.yml"))
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad, h = stream_grad_and_hess_win(data, synt, [dz, ddt])

    return cost, grad, np.outer(h, h)


# depths = np.arange(cmt_goal.depth_in_m - 10000,
#                    cmt_goal.depth_in_m + 10100, 1000) / 1000.0
# times = np.arange(-10.0, 10.1, 1.0)
# t, z = np.meshgrid(times, depths)
# cost = np.zeros(z.shape)
# grad = np.zeros((*z.shape, 2))
# hess = np.zeros((*z.shape, 2, 2))
# dm = np.zeros((*z.shape, 2))
# for _i, _dep in enumerate(depths):
#     for _j, _time in enumerate(times):
#         print_action(f"Computing depth: ({_dep} km, {_time} s)")
#         c, g, h = compute_cost_and_gradient_hessian2(np.array([_dep, _time]))
#         cost[_i, _j] = c
#         grad[_i, _j, :] = g
#         hess[_i, _j, :, :] = h

# damp = 0.0001
# # Get the Gauss newton step
# for _i in range(z.shape[0]):
#     for _j in range(z.shape[1]):
#         dm[_i, _j, :] = np.linalg.solve(
#             hess[_i, _j, :, :] + damp * np.diag(np.ones(2)), - grad[_i, _j, :])


# def plot_label(ax: matplotlib.axes.Axes, label: str, aspect: float = 1,
#                location: int = 1, dist: float = 0.025, box: bool = True,
#                fontdict: dict = {}):
#     """Plots label one of the corners of the plot.

#     .. code::

#         1-----2
#         |     |
#         3-----4


#     Parameters
#     ----------
#     label : str
#         label
#     aspect : float, optional
#         aspect ratio length/height, by default 1.0
#     location : int, optional
#         corner as described by above code figure, by default 1
#     aspect : float, optional
#         aspect ratio length/height, by default 0.025
#     box : bool
#         plots bounding box st. the label is on a background, default true
#     """
#     if box:
#         box = {'facecolor': 'w', 'edgecolor': 'k'}
#     else:
#         box = {}

#     if location == 1:
#         plt.text(dist * aspect, 1.0 - dist, label, horizontalalignment='left',
#                  verticalalignment='top', transform=ax.transAxes, bbox=box,
#                  fontdict=fontdict)
#     elif location == 2:
#         plt.text(1.0 - dist * aspect, 1.0 - dist, label,
#                  horizontalalignment='right', verticalalignment='top',
#                  transform=ax.transAxes, bbox=box,
#                  fontdict=fontdict)
#     elif location == 3:
#         plt.text(dist * aspect, dist, label, horizontalalignment='left',
#                  verticalalignment='bottom', transform=ax.transAxes, bbox=box,
#                  fontdict=fontdict)
#     elif location == 4:
#         plt.text(1.0 - dist * aspect, dist, label,
#                  horizontalalignment='right', verticalalignment='bottom',
#                  transform=ax.transAxes, bbox=box,
#                  fontdict=fontdict)
#     else:
#         raise ValueError("Other corners not defined.")


# fontdict = {'fontsize': 7}
# extent = [np.min(t), np.max(t), np.min(z), np.max(z)]
# aspect = (np.max(t) - np.min(t))/(np.max(z) - np.min(z))
# plt.figure(figsize=(11, 6.5))
# # Cost
# ax1 = plt.subplot(3, 4, 9)
# plt.imshow(cost, interpolation=None, extent=extent, aspect=aspect)
# plot_label(ax1, r"$\mathcal{C}$", dist=0)
# c1 = plt.colorbar()
# c1.ax.tick_params(labelsize=7)
# c1.ax.yaxis.offsetText.set_fontsize(7)
# ax1.axes.invert_yaxis()
# plt.ylabel(r'$z$')
# plt.xlabel(r'$t$')

# # Gradient
# ax2 = plt.subplot(3, 4, 6, sharey=ax1)
# plt.imshow(grad[:, :, 1], interpolation=None, extent=extent, aspect=aspect)
# c2 = plt.colorbar()
# c2.ax.tick_params(labelsize=7)
# c2.ax.yaxis.offsetText.set_fontsize(7)
# ax2.tick_params(labelbottom=False)
# plot_label(ax2, r"$g_{\Delta t}$", dist=0)

# ax3 = plt.subplot(3, 4, 10, sharey=ax1)
# plt.imshow(grad[:, :, 0], interpolation=None, extent=extent, aspect=aspect)
# c3 = plt.colorbar()
# c3.ax.tick_params(labelsize=7)
# c3.ax.yaxis.offsetText.set_fontsize(7)
# ax3.tick_params(labelleft=False)
# plot_label(ax3, r"$g_z$", dist=0)
# plt.xlabel(r'$\Delta t$')

# # Hessian
# ax4 = plt.subplot(3, 4, 3, sharey=ax1)
# plt.imshow(hess[:, :, 0, 1], interpolation=None, extent=extent, aspect=aspect)
# c4 = plt.colorbar()
# c4.ax.tick_params(labelsize=7)
# c4.ax.yaxis.offsetText.set_fontsize(7)
# ax4.tick_params(labelbottom=False)
# plot_label(ax4, r"$\mathcal{H}_{z,\Delta t}$", dist=0)

# ax5 = plt.subplot(3, 4, 7, sharey=ax1)
# plt.imshow(hess[:, :, 1, 1], interpolation=None, extent=extent, aspect=aspect)
# c5 = plt.colorbar()
# c5.ax.tick_params(labelsize=7)
# c5.ax.yaxis.offsetText.set_fontsize(7)
# ax5.tick_params(labelleft=False, labelbottom=False)
# plot_label(ax5, r"$\mathcal{H}_{\Delta t,\Delta t}$", dist=0)

# ax6 = plt.subplot(3, 4, 11, sharey=ax1)
# plt.imshow(hess[:, :, 0, 0], interpolation=None, extent=extent, aspect=aspect)
# c6 = plt.colorbar()
# c6.ax.tick_params(labelsize=7)
# c6.ax.yaxis.offsetText.set_fontsize(7)
# ax6.tick_params(labelleft=False)
# plot_label(ax6, r"$\mathcal{H}_{z,z}$", dist=0)
# plt.xlabel(r'$\Delta t$')

# # Gradient/Hessian
# ax7 = plt.subplot(3, 4, 8, sharey=ax1)
# plt.imshow(dm[:, :, 1], interpolation=None, extent=extent, aspect=aspect)
# c7 = plt.colorbar()
# c7.ax.tick_params(labelsize=7)
# c7.ax.yaxis.offsetText.set_fontsize(7)
# ax7.tick_params(labelleft=False, labelbottom=False)
# plot_label(ax7, r"$\mathrm{d}\Delta$", dist=0)

# ax8 = plt.subplot(3, 4, 12, sharey=ax1)
# plt.imshow(dm[:, :, 0], interpolation=None, extent=extent, aspect=aspect)
# c8 = plt.colorbar()
# c8.ax.tick_params(labelsize=7)
# c8.ax.yaxis.offsetText.set_fontsize(7)
# ax8.tick_params(labelleft=False)
# plot_label(ax8, r"$\mathrm{d}z$", dist=0)
# plt.xlabel(r'$\Delta t$')

# plt.subplots_adjust(hspace=0.2, wspace=0.15)
# plt.savefig(f"SyntheticCostGradHess{eventype}.pdf")


# Define initial model that is 10km off
model = np.array([(cmt_goal.depth_in_m + 10000.0)/1000.0, - 5.0])

print_section("BFGS")
# Prepare optim steepest
optim = Optimization("bfgs")
optim.compute_cost_and_gradient = compute_cost_and_gradient2
optim.is_preco = False
optim.niter_max = 7
optim.stopping_criterion = 1e-8
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

plot_optimization(
    optim_bfgs, outfile=f"SyntheticDepthInversionMisfitReduction{eventype}.pdf")
plot_model_history(optim_bfgs, labellist=[r'$z$', r'$\Delta t$'],
                   outfile=f"SyntheticDepthInversionModelHistory{eventype}.pdf")

print_section("GN")
# Prepare optim steepest
optim = Optimization("gn")
optim.compute_cost_and_grad_and_hess = compute_cost_and_gradient_hessian2
optim.is_preco = False
optim.niter_max = 7
optim.damping = 0.1
optim.stopping_criterion = 1e-8
optim.n = len(model)
optim_gn = optim.solve(optim, model)

plot_optimization(
    [optim_bfgs, optim_gn],
    outfile=f"SyntheticDepthInversionMisfitReduction{eventype}.pdf")
plot_model_history([optim_bfgs, optim_gn], labellist=[r'$z$', r'$\Delta t$'],
                   outfile=f"SyntheticDepthInversionModelHistory{eventype}.pdf")

ax = plot_station_xml(station_xml)
ax.add_collection(
    beach(cmt_init.tensor, xy=(cmt_init.longitude, cmt_init.latitude),
          width=5, size=100, linewidth=1.0))
plt.savefig("SyntheticAcquisition.pdf")
