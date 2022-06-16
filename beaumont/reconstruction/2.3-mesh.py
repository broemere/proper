# @author: broemere
# created: 6/3/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("3-volume")

savedir = outputdir / "mesh-renders"
mkdirpy(savedir)

export = True

for s in dd.samples:
    print(s.letter)
    s.videoframes = []
    s.meshes = []
    s.mesherrors = []
    #del c.framelocs[0], c.framelocs[-1]

    s.intervals0 = deepcopy(s.intervals)

    #s.framelocs.remove(s.intervals0[0])
    #s.framelocs.remove(s.intervals0[-1])

    for cube2, i in zip(s.cubes2, s.intervals):
        holesfilled = False
        verts, faces, normals, values = measure.marching_cubes(cube2, 1, gradient_direction="ascent", step_size=2,
                                                               allow_degenerate=False)
        tmesh = trimesh.Trimesh(verts, faces)
        if not tmesh.is_watertight:
            print("Error: " + str(i) + " is not watertight")
            tmesh.fill_holes()
            holesfilled = True
        tmesh = trimesh.smoothing.filter_laplacian(tmesh, lamb=0.5, iterations=10, implicit_time_integration=False,
                                                   volume_constraint=True, laplacian_operator=None)
        tmesh.apply_scale(0.064)
        if not tmesh.is_watertight:
            print("Watertight error")
            s.mesherrors.append(i)
        if holesfilled:
            if tmesh.is_watertight:
                print("Watertight successfully repaired")
        if tmesh.euler_number != 2:
            print("Euler error")
            s.mesherrors.append(i)
        #print(f"\tVolume: {round(tmesh.volume, 3)}")

        scene = trimesh.Scene()
        rot1 = trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 1, 0], point=tmesh.centroid)
        rot2 = trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 0, 1], point=tmesh.centroid)
        rot3 = trimesh.transformations.rotation_matrix(np.deg2rad(180), [0, 1, 0], point=tmesh.centroid)
        # newcenter = np.round(np.array(tmesh.centroid))
        translation = -tmesh.centroid
        tmesh.apply_translation(translation)
        tmesh.apply_transform(rot1)
        tmesh.apply_transform(rot2)
        tmesh.apply_transform(rot3)

        scene.add_geometry(tmesh)
        img = scene.save_image((s.height, s.height))
        imgarr = np.frombuffer(img, np.uint8)
        imgdc = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
        #imgdc = cv2.cvtColor(imgdc, cv2.COLOR_BGR2GRAY) # No way to display single channel gray. Always turns black
        s.videoframes.append(imgdc)
        #display(imgdc, True)
        s.meshes.append(tmesh)


timer.check(True)

writeinterm("2-mesh", dd)