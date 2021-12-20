"""
This script generates a figure that compares different damping styles.

Full diagonal vs. only hypocenter vs. full only for shallow events and otherwise
hypocenter only.

It uses the original GCMT catalog, Hessians, gradients, and scaling used 
as a part of the GCMT3D study.

"""

import os
import matplotlib.pyplot as plt
from lwsspy.gcmt3d.plot_inversion_distributions import get_stuff
from lwsspy.gcmt3d.plot_inversion_distributions import inversion_tests, inversion_tests_depth
from lwsspy.gcmt3d.plot_inversion_distributions import plot_results, plot_results_depth

plt.switch_backend('pdf')

# Dirs
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
figdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files
catfile = os.path.join(datadir, 'gcmtcatalog.pkl')
inversionfile = os.path.join(datadir, 'inversion.npz')
scalingfile = os.path.join(datadir, 'scaling.txt')
outfile = os.path.join(figdir, 'damping_mt.svg')

# Load all necessary parameters
# G are the gradients, H are the Hessians, scaling is the scaling file.
events, G, H, scaling, cat = get_stuff(
    inversionfile, scalingfile, catfile)

# Compute different dampings for each style of damping
dms = []
dampstyles = ['all', 'depth', 'hypo']
for _dtype in dampstyles:
    _dms, dampinglist = inversion_tests(
        events, G, H, scaling, cat, damp_type=_dtype)
    dms.append(_dms)

# Plot the results
plot_results(dms, dampinglist, cat, titles=dampstyles)

# Save the plot
plt.savefig(outfile)
