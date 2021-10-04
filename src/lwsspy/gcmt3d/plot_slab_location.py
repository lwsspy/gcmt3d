# External
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from obspy.imaging.beachball import beach
from cartopy import crs
from obspy.geodetics.base import gps2dist_azimuth
# Internal
from .. import plot as lplt
from .. import geo as lgeo
from .. import maps as lmap
from .. import seismo as lseis


def plot_slab_location(cmtsource: lseis.CMTSource, cmtsource2=None,
                       point: bool = False):

    # Get slab location
    dss = lgeo.get_slabs()
    vmin, vmax = lgeo.get_slab_minmax(dss)

    # Compute levels
    levels = np.linspace(vmin, vmax, 200)

    # Define color mapping
    cmap = plt.get_cmap('rainbow')
    norm = BoundaryNorm(levels, cmap.N)

    # Filter slab datasets (don't need to plot all of them)
    filtered_dss = []

    # Get extent in which to include surrounding slabs
    lat = cmtsource.latitude
    lon = cmtsource.longitude
    buffer = 10
    buffer05 = 0.1 * buffer
    extent = lon - buffer, lon + buffer, lat - buffer, lat + buffer
    extent05 = lon - buffer05, lon + buffer05, lat - buffer05, lat + buffer05

    for ds in dss:

        # Get slab coordinates
        llon, llat = np.meshgrid(ds['x'][:].data, ds['y'][:].data)
        llon = np.where(llon > 180.0, llon - 360, llon)

        # Check if slab overlaps
        if lmap.in_extent(*extent, llon, llat):
            filtered_dss.append(ds)

    # Plot map
    proj = crs.Mollweide(central_longitude=lon)
    ax = plt.axes(projection=proj)
    # ax.set_global()
    lmap.plot_map()
    lmap.plot_map(fill=False, borders=False, zorder=10)
    ax.set_extent(extent05)

    # Plot map with central longitude on event longitude
    lgeo.plot_slabs(dss=filtered_dss, levels=levels, cmap=cmap, norm=norm)

    # Plot CMT as popint or beachball
    cdepth = cmap(norm(cmtsource.depth_in_m/-1000.0))

    if point:
        plt.plot(lon, lat, 's', markeredgecolor='k', markerfacecolor=cdepth,
                 linestyle='-', linewidth=1.0, zorder=100,
                 transform=crs.PlateCarree())
    else:

        # Transform points because beachball isn't in map coordinates
        p = lmap.geo2disp(proj, np.array(lon), np.array(lat))
        x, y = p[0, 0], p[0, 1]

        # Plot CMT 1
        b = beach(cmtsource.tensor, facecolor=cdepth, width=30,
                  xy=(x, y), zorder=100, linewidth=0.75, alpha=1.0,
                  edgecolor='k', axes=ax, size=100, nofill=False)
        ax.add_collection(b)

    if cmtsource2 is not None:
        cdepth = cmap(norm(cmtsource.depth_in_m/-1000.0))
        gc, az, baz = gps2dist_azimuth(
            lat, lon, cmtsource2.latitude, cmtsource2.longitude)
        gc /= 1000.0  # --> km
        gc /= 111.11  # --> deg

        # Put the c
        plt.title(f"dx: {gc:5.2f} deg -- az: {az:6.2f} deg")
        if point:
            plt.plot(
                cmtsource2.longitude, cmtsource2.latitude, 'o',
                markeredgecolor='k', markerfacecolor=cdepth,
                linewidth=0.5, zorder=101,
                transform=crs.PlateCarree())
        else:
            p = lmap.geo2disp(
                proj,
                np.array(cmtsource2.longitude),
                np.array(cmtsource2.latitude))
            x, y = p[0, 0], p[0, 1]

            # Plot CMT 2
            cdepth = cmap(norm(cmtsource.depth_in_m/-1000.0))
            b = beach(cmtsource.tensor, facecolor=cdepth, width=30,
                      xy=(x, y), zorder=101, linewidth=0.25, alpha=1.0,
                      edgecolor='k', axes=ax, size=100, nofill=False)
            ax.add_collection(b)

    c = lplt.nice_colorbar(aspect=40, fraction=0.1, shrink=0.6)
    c.set_label("Depth [km]")
    axins = plt.axes(
        [0.02, 0.75, 0.23, 0.23],
        projection=crs.Orthographic(
            central_longitude=lon, central_latitude=lat))
    axins.set_global()
    lmap.plot_map()
    axins.plot(lon, lat, '*r', markeredgecolor='k')

    return cmap, norm
