# @author: broemere
# created: 6/13/2022
"""
~~~Description~~~
"""
from elib import *
from mayavi import mlab
timer = runstartup()
dd = loadinterm("2-vox-parallel")

mlab.options.offscreen = True

export = True
savedir = outputdir / "voxel-renders"
mkdirpy(savedir)
framedir = intermdir / "movie-frames"
mkdirpy(framedir)
moviedir = framedir / "voxel-rotation"
mkdirpy(moviedir)

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1080, 1080))

for s in dd.samples:
    print(s.letter)
    sampledir = moviedir / f"last-{s.letter}"
    mkdirpy(sampledir)
    coords = s.cubes2[-1].nonzero()

    for i in range(50):
        mlab.points3d(coords[0], coords[1], coords[2], mode="cube", color=(0.8,1,1), scale_factor=1)
        mlab.view(azimuth=-10-(i*7.2), elevation=70)
        mlab.orientation_axes()
        num = str(i).zfill(2)
        filename = f"{s.letter}{num}.png"
        filepath = sampledir / filename
        mlab.savefig(str(filepath), magnification=3)
        #mlab.show()
        mlab.clf()

    mlab.clf()

    timer.total(True)

mlab.close()

timer.total(True)