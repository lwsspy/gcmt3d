# %%
import os
from cartopy.crs import PlateCarree
import matplotlib.pyplot as plt
from scipy.spatial import kdtree
from lwsspy.geo.plot_slab_slices import plot_slab_slices
from lwsspy.math.SphericalNN3D import SphericalNN3D
from lwsspy.seismo.cmt_catalog import CMTCatalog
from lwsspy.maps.plot_line_buffer import plot_line_buffer
from lwsspy.maps.scalebar import scale_bar

import numpy as np


def plot_slab_cmts(slice_dict):

    # %% Load CMT catalog

    oldcat = CMTCatalog.load(os.path.join(
        '/Users/lucassawade/stats/catalogs', 'gcmt.pkl'))
    newcat = CMTCatalog.load(os.path.join(
        '/Users/lucassawade/stats/catalogs', 'gcmt3d+_fix.pkl'))

    oldcat = oldcat.in_catalog(['C201811012219A'])
    ocat, ncat = oldcat.check_ids(newcat)
    # %% find nearest neighbours

    latitude = ocat.getvals('latitude')
    longitude = ocat.getvals('longitude')
    depth = ocat.getvals('depth_in_m')/1000.0

    nlatitude = ncat.getvals('latitude')
    nlongitude = ncat.getvals('longitude')
    ndepth = ncat.getvals('depth_in_m')/1000.0

    # %%
    # Make nan filled arrays for simple line plotting
    intermlon = np.vstack(
        (longitude, nlongitude, np.nan * np.ones_like(longitude))).T.flatten()
    intermlat = np.vstack(
        (latitude, nlatitude, np.nan * np.ones_like(longitude))).T.flatten()

    # %%
    tracks, mapax, axes, skt, slablist = plot_slab_slices(
        slice_dict=slice_dict)
    z = np.arange(0, 800, 1)
    maxdist = 1.5

    # %%
    # Text and line can be styled separately. Keywords are simply passed to
    # text or plot.
    text_kwargs = dict(family='sans', size='medium', color='k')
    plot_kwargs = dict(linestyle='solid', color='k', linewidth=3.0)
    # scale_bar(mapax, (0.0, 0.5), 500, metres_per_unit=1000, unit_name='km',
    #           text_kwargs=text_kwargs,
    #           plot_kwargs=plot_kwargs)

    # %%
    counter = 0
    kdtrees = []
    shapes = []
    for _track in tracks:

        lats, lons, dists = _track

        llon, ddep = np.meshgrid(lons, z)
        llat, _ = np.meshgrid(lats, z)

        shapes.append(llat.shape)

        kdtrees.append(SphericalNN3D(
            llat.flatten(), llon.flatten(), ddep.flatten()))

        counter += 1

    # %%
    iscatterkwargs = dict(
        marker='o',
        markersize=1.5,
        markerfacecolor='k',
        markeredgecolor='k',
        markeredgewidth=0.2,
        linestyle='none'
    )
    fscatterkwargs = dict(
        marker='o',
        markersize=2.25,
        markerfacecolor='w',
        markeredgecolor='k',
        markeredgewidth=0.2,
        linestyle='none'
    )

    for _i, (_tree, _shape, _track) in enumerate(zip(kdtrees, shapes, tracks)):

        lats, lons, dists = _track
        plt.sca(mapax)

        plot_line_buffer(lats, np.where(lons < 0.0, lons + 360, lons),
                         c180=slice_dict['c180'], delta=maxdist, ax=mapax,
                         linestyle='--', linewidth=0.5, alpha=1.0,
                         facecolor='none', edgecolor='k',
                         zorder=20, transform=PlateCarree())

        # %% Query the kdtree
        # i are the indeces in the kdtree, pos are the indeces in the geovector,
        # d are the distances between them
        d, i, pos = _tree.query(latitude, longitude, qd=depth,
                                maximum_distance=maxdist)
        nd, ni, npos = _tree.query(nlatitude[pos], nlongitude[pos], qd=ndepth[pos],
                                   maximum_distance=2*maxdist)

        idx = np.unravel_index(i, _shape)
        zidx = idx[0]
        didx = idx[1]

        nidx = np.unravel_index(ni, _shape)
        nzidx = nidx[0]
        ndidx = nidx[1]

        axes[_i].plot(
            np.vstack((dists[didx], dists[ndidx])),
            np.vstack((z[zidx], z[nzidx])),
            'k', lw=0.5
        )
        axes[_i].plot(dists[didx], z[zidx], **iscatterkwargs)
        axes[_i].plot(dists[ndidx], z[nzidx], **fscatterkwargs)

    mapax.plot(intermlon, intermlat,
               'k', lw=0.5, transform=PlateCarree())
    mapax.plot(longitude, latitude, **iscatterkwargs,
               transform=PlateCarree())
    mapax.plot(nlongitude, nlatitude, **fscatterkwargs,
               transform=PlateCarree())
