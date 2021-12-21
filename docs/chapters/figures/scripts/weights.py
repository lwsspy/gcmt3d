"""
This script plots the final weight distribution for a specific event.

Here: C201811012219A

This is Figure 3 in the paper Sawade et al. (2022)

"""
import os
import lwsspy.plot as lplt
from lwsspy.gcmt3d.plot_weights import plot_final_weight_pickle
import matplotlib.pyplot as plt

# Plot config
lplt.updaterc()
plt.switch_backend('pdf')

# Dirs
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
figdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Weight pickle file
wpf = os.path.join(datadir, 'C201811012219A_weights.pkl')

# Plt the weights
plot_final_weight_pickle(wpf)
plt.suptitle('C201811012219A')

# Save figures in 3 formats
plt.savefig(os.path.join(figdir, "weights.pdf"),
            transparent=True)
plt.savefig(os.path.join(figdir, "weights.svg"),
            transparent=True)
plt.savefig(os.path.join(figdir, "weights.png"), dpi=300,
            transparent=True)
