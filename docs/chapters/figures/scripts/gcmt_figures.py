"""

Script generates some figures of the GCMT catalog.

files created (with both endings ``.png`` and ``.svg``):

- gcmt_depth_moment
- gcmt_depth_moment_compare
- gcmt_depth_moment_compare_side

"""

import os
from copy import deepcopy
import lwsspy.plot as lplt
import lwsspy.seismo as lseis
import lwsspy.maps as lmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.ticker import StrMethodFormatter
import cartopy
from cartopy.crs import PlateCarree, Mollweide
import numpy as np


# Plot config
lplt.updaterc()
# plt.switch_backend('svg')

# Dirs
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
figdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get catalogs
oldcat = lseis.CMTCatalog.load(os.path.join(
    datadir, 'gcmtcatalog.pkl'))

latitude = oldcat.getvals(vtype='latitude')
longitude = oldcat.getvals(vtype='longitude')
depth = oldcat.getvals(vtype='depth_in_m')/1000.0
moment = oldcat.getvals(vtype='moment_magnitude')

###########################   SCATTER #########################################

# Create png
_, ax, _, _ = lseis.plot_quakes(latitude, longitude, depth, moment,
                                cmap='rainbow_r', yoffsetlegend=0.05)
outnamepng = os.path.join(figdir, "gcmt_depth_moment.png")
plt.savefig(outnamepng, transparent=True, dpi=300)
plt.close()

# Create vectors
_, ax, _, _ = lseis.plot_quakes(latitude, longitude, depth, moment,
                                cmap='rainbow_r', yoffsetlegend=0.05)
outnamesvg = os.path.join(figdir, "gcmt_depth_moment.svg")
plt.savefig(outnamesvg, transparent=True)

# Scatter submap
_, ax, _, _ = lseis.plot_quakes(latitude, longitude, depth, moment,
                                cmap='rainbow_r', yoffsetlegend=0.0)
fig = ax.figure
fig.set_size_inches(6, 10)
plt.subplots_adjust(left=0.025, right=0.975, bottom=0.03, top=1.0)
ax.set_extent([160, 185, -50, -10])

# Save PNG
outnamepng = os.path.join(figdir, "gcmt_depth_moment_tonga.png")
plt.savefig(outnamepng, dpi=300, transparent=True)

# Save SVG
outnamesvg = os.path.join(figdir, "gcmt_depth_moment_tonga.svg")
plt.savefig(outnamesvg, transparent=True)
plt.close()


##############################################################################
lsel = 5.5
hsel = 7.5
selection = (lsel <= moment) & (moment <= hsel)

rmoment = deepcopy(moment)
rmoment[hsel <= rmoment] = hsel
rmoment[rmoment <= lsel] = lsel


plt.figure(figsize=(5, 6.5))
plt.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.95)
ax1 = plt.subplot(211, projection=Mollweide(central_longitude=-150.0))
ax1.set_global()
lmap.plot_map(zorder=-1)
lseis.plot_quakes(latitude, longitude, depth, rmoment, ax=ax1,
                  cmap='rainbow_r', legend=False)
lplt.plot_label(ax1, "a)", box=False)
lplt.plot_label(ax1, f"N: {len(depth)}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title("GCMT Events")
ax2 = plt.subplot(212, projection=Mollweide(central_longitude=-150.0))
ax2.set_global()
lmap.plot_map(zorder=-1)
lseis.plot_quakes(latitude[selection], longitude[selection],
                  depth[selection], moment[selection], ax=ax2,
                  cmap='rainbow_r')
lplt.plot_label(ax2, "b)", box=False)
lplt.plot_label(ax2, f"N: {len(depth[selection])}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title(f"GCMT Events {lsel} $\leq M_w \leq$ {hsel}")


# Save as PNG
outnamepng = os.path.join(figdir, "gcmt_depth_moment_compare.png")
plt.savefig(outnamepng, dpi=300)

# Save as SVG
outnamesvg = os.path.join(figdir, "gcmt_depth_moment_compare.svg")
plt.savefig(outnamesvg, transparent=True)

# Sideways ##################################################################


plt.figure(figsize=(10, 3.5))
plt.subplots_adjust(left=0.025, right=0.975,
                    bottom=0.15, top=0.95, wspace=0.05)
ax1 = plt.subplot(121, projection=Mollweide(central_longitude=-150.0))
ax1.set_global()
lmap.plot_map(zorder=-1)
lseis.plot_quakes(latitude, longitude, depth, rmoment, ax=ax1,
                  cmap='rainbow_r', legend=True, xoffsetlegend=0.525)
lplt.plot_label(ax1, "a)", box=False)
lplt.plot_label(ax1, f"N: {len(depth)}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title("GCMT Events")
ax2 = plt.subplot(122, projection=Mollweide(central_longitude=-150.0))
ax2.set_global()
lmap.plot_map(zorder=-1)
scatter, ax, l1, l2 = lseis.plot_quakes(
    latitude[selection], longitude[selection],
    depth[selection], moment[selection], ax=ax2,
    cmap='rainbow_r', legend=False)

lplt.plot_label(ax2, "b)", box=False)
lplt.plot_label(ax2, f"N: {len(depth[selection])}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))

plt.title(f"GCMT Events {lsel} $\leq M_w \leq$ {hsel}")

# Save as PDF
outnamepng = os.path.join(figdir, "gcmt_depth_moment_compare_side.png")
plt.savefig(outnamepng, transparent=True, dpi=300)

# Save as PDF
outnamesvg = os.path.join(figdir, "gcmt_depth_moment_compare_side.svg")
plt.savefig(outnamesvg, transparent=True)
