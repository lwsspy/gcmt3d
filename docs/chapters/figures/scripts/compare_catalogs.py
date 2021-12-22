import os
from lwsspy.plot import updaterc
import matplotlib.pyplot as plt
from lwsspy.seismo.cmt_catalog import CMTCatalog
import lwsspy.seismo as lseis

# Plot config
updaterc()
plt.switch_backend('pdf')


# Dirs
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
figdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get catalogs
initcat = CMTCatalog.load(os.path.join(
    datadir, 'gcmt.pkl'))
newcat = CMTCatalog.load(os.path.join(
    datadir, 'gcmt3d+_fix.pkl'))
initcat, newcat = initcat.check_ids(newcat)


# Create comparison class
C = lseis.CompareCatalogs(
    initcat, newcat, oldlabel='GCMT', newlabel='GCMT3D+X')
Cfilt = C.filter(maxdict=dict(depth_in_m=30000.0, M0=.5))[0]
Cfilt.plot_summary()

# Compare two catalogs
plt.savefig(os.path.join(figdir, 'catalog_comparison.svg'), transparent=True)

# Spatial relocation
Cfilt.plot_spatial_distribution("location")
plt.savefig(os.path.join(figdir, "spatial_relocation.svg"), transparent=True)
