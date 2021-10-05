from lwsspy.seismo.invertcmt.plot_slab_cmts import plot_slab_cmts

# %%

south_america_shallow_dict = dict(
    # Some point at the coast
    olat=-22.336766,
    olon=-70.042413,
    lat0=[2.5, -6.5, -16.8, -23.8],
    lon0=[-78.5, -81.7, -74.52, -71.0],
    lat1=[2.5, -4.8, -14.5, -22.6],
    lon1=[-72.5, -74.8, -67.4, -65.0],
    xlim=[0, 10],
    ylim=[0, 400],
    dist=0.05,
    mapextent=[-87.5, -47.5, -55, 20],
    caxextent=[0.15, 0.0, 0.03, 0.3],
    central_longitude=0.0,
    invertx=False,
    c180=False
)

plot_slab_cmts(south_america_shallow_dict)

# %%

south_america_deep_dict = dict(
    # Some point at the coast
    olat=-22.336766,
    olon=-70.042413,
    lat0=[-9.3,  -20.6, -29.6],
    lon0=[-76.8, -69.0, -68.8],
    lat1=[-8.1, -18.9, -26.9],
    lon1=[-69.0, -61.2, -60.3],
    xlim=[0, 10],
    ylim=[200, 750],
    dist=0.0025,
    mapextent=[-87.5, -47.5, -55, 20],
    caxextent=[0.15, 0.0, 0.03, 0.3],
    central_longitude=0.0,
    invertx=False,
    c180=False
)

plot_slab_cmts(south_america_deep_dict)

# %%

tonga_dict = dict(
    # Some point at the coast
    olat=-22.336766,
    olon=-180.042413,
    lat0=[-15.4, -20.5, -26.0, -29.56],
    lon0=[-173.0+360, -173.75+360, -176.0+360, -176.6+360],
    lat1=[-17.35, -20.0, -25.0, -30.753],
    lon1=[-179.0+360, -178.5+360, -179.9+360, 179.84],
    xlim=[0, 6.5],
    ylim=[0, 400],
    dist=0.0025,
    mapextent=[167.5, -170 + 360, -50, -13],
    caxextent=[1-0.15, 0.0, 0.03, 0.3],
    central_longitude=180.0,
    invertx=True,
    c180=True
)

plot_slab_cmts(tonga_dict)

# %%
tonga_deep_dict = dict(
    # Some point at the coast
    olat=-22.336766,
    olon=-180.042413,
    lat0=[-16.8, -20.5, -25.0],
    lon0=[-177.0+360, -176.8+360, -179.0+360],
    lat1=[-18.5, -20.5, -23.875],
    lon1=[179.0, 179.5, 177.9],
    xlim=[0, 4.5],
    ylim=[300, 700],
    dist=0.0025,
    mapextent=[170, -170 + 360, -50, -13],
    caxextent=[1-0.15, 0.0, 0.03, 0.3],
    central_longitude=180.0,
    invertx=True,
    c180=True
)


plot_slab_cmts(tonga_deep_dict)
