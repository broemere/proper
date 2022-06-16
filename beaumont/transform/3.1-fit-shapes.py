# @author: broemere
# created: 6/9/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("2-vox-parallel")

scale = 0.064

for s in dd.samples:  # Derive r from actual volume
    s.p = s.pt["p"].iloc[s.intervals]
    s.t = s.pt["t"].iloc[s.intervals]
    s.vols = np.array(s.vols)
    s.rs = (s.vols*(3/4)*(1/math.pi))**(1/3)


for s in dd.samples:  # Get r from fitted sphere and ellipse
    print(s.letter)
    s.spherers = []
    s.da = []
    s.db = []
    s.dc = []
    for i, cube2 in enumerate(s.cubes2):
        if i % 33 == 0:
            print(f"{i}%")
        cubecoords = cube2.nonzero()
        x, y, z, r = ls_sphere(cubecoords[0], cubecoords[1], cubecoords[2])
        s.spherers.append(r*scale)
        eansa = ls_ellipsoid(cubecoords[0], cubecoords[1], cubecoords[2])
        center, axes, inve = polyToParams3D(eansa, False)
        s.da.append(axes[0]*scale)
        s.db.append(axes[1]*scale)
        s.dc.append(axes[2]*scale)

    s.spherers = np.array(s.spherers)
    s.da = np.array(s.da)
    s.db = np.array(s.db)
    s.dc = np.array(s.dc)

    del s.cubes2


timer.check(True)
writeinterm("3-shapes", dd)

