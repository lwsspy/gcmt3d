# %%
import os
import lwsspy as lpy
# Load Catalogs

initcat = lpy.seismo.CMTCatalog.load(
    os.path.join('/Users/lucassawade', 'oldcat.pkl'))
newcat = lpy.seismo.CMTCatalog.load(
    os.path.join('/Users/lucassawade', 'newcat.pkl'))

newcat, newp = newcat.filter(mindict=dict(depth_in_m=0.0))

newp.plot()

# %%

print("Old:", len(initcat.cmts))
print("New:", len(newcat.cmts))

# Get overlaps
ocat, ncat = initcat.check_ids(newcat)

print("After checkid:")
print("  Old:", len(ocat.cmts))
print("  New:", len(ncat.cmts))

# %%

# Compare Catalog
CC = lpy.seismo.CompareCatalogs(old=ocat, new=ncat,
                                oldlabel='GCMT', newlabel='GCMT3D+X',
                                nbins=25)

# %%

_, M0_pop = CC.filter(
    maxdict={"M0": 0.5, })
_, z_pop = CC.filter(
    maxdict={"depth_in_m": 25000.0})

_, loc_pop = CC.filter(
    maxdict={"latitude": 0.35,
             "longitude": 0.35})

_, ts_pop = CC.filter(
    maxdict={"time_shift": 10.0})


# %%
M0_pop.plot_summary()
z_pop.plot_summary()
loc_pop.plot_summary()
ts_pop.plot_summary()
# CC_pop.plot_summary()

# %%

loc_pop.plot_spatial_distribution(parameter='location')
# CC_pop.plot_spatial_distribution(parameter='depth_in_m')
