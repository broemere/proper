# @author: broemere
# created: 6/3/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("2-ocf-parallel")


for s in dd.samples:  # 3 min per 100
    s = carve(s)
    s = hollow(s)
    del s.cubes


timer.check(True)
nsdetails(s)

writeinterm("2-vox", dd)


#export = True
#savedir = outputdir / "voxel-renders"
#mkdirpy(savedir)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(c2coords[0], c2coords[1], c2coords[2])
        # #ax.set_xlim(0, max(x, y, z))  # a = 6 (times two for 2nd ellipsoid)
        # #ax.set_ylim(0, max(x, y, z))  # b = 10
        # #ax.set_zlim(0, max(x, y, z))  # c = 16
        # plt.tight_layout()
        # plt.show()

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.gca(projection='3d')
        # ax.voxels(cube2)
        # plt.tight_layout()
        # plt.show()


# c.cubesavg = []
# c.avgvols = []
#
# for i in np.linspace(1, len(c.cubes)-2, len(c.cubes)-2):
#     i = int(i)
#     c1 = c.cubes[i-1]
#     c2 = c.cubes[i]
#     c3 = c.cubes[i+1]
#     ctot = c1 + c2 + c3
#     cavg = np.round(ctot/3)
#     c.cubesavg.append(cavg.astype(np.uint8))
#     vol = round(sum(sum(sum(cavg))) * 0.064 * 0.064 * 0.064, 3)
#     c.avgvols.append(vol)

# exportn = 0
# if export:
#     for cube, i in zip(s.cubesavg, range(len(s.cubesavg))):
#         if i == exportn + 1:
#             x = cube.shape[0]
#             y = cube.shape[1]
#             z = cube.shape[2]
#             x1 = np.linspace(0, x - 1, x)
#             y1 = np.linspace(0, y - 1, y)
#             z1 = np.linspace(0, z - 1, z)
#             X, Y, Z = np.meshgrid(y1, x1, z1)
#             fig = go.Figure(data=go.Volume(
#                 x=X.flatten(),
#                 y=Y.flatten(),
#                 z=Z.flatten(),
#                 value=cube.flatten(),
#                 opacity=1,
#                 isomin=0.1,
#                 colorscale='tealgrn_r'))
#             filename = s._label + "Volume_" + str(i) + ".html"
#             fig.write_html(str(savedir / filename))