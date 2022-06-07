# @author: broemere
# created: 6/3/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("2-ocf")

savedir = outputdir / "voxel-renders"
mkdirpy(savedir)

usemiddle = True
export = True

for s in dd.samples:  # 3 min per 100
    print(s.letter)
    s.vols = []
    s.cubes = []
    for i in range(len(s.intervals)):
        if i % 33 == 0:
            print(i)
        if "top" in s.redviews:
            v = getview(s, "top")
            top = v.fills[i]
            top = cv2.flip(top, 0)
        else:
            v = getview(s, "bottom")
            top = v.fills[i]
        if "left" in s.redviews:
            v = getview(s, "left")
            left = v.fills[i]
            left = cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            v = getview(s, "right")
            left = v.fills[i]
            left = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)

        maxy = max(left.shape[0], top.shape[0])
        leftx = left.shape[1]
        topx = top.shape[1]

        # Realign tops of bladder views and re-pad blank space
        y_nonzero, x_nonzero = np.nonzero(top)
        # top = top[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        top = top[:np.max(y_nonzero) + 1, :]
        y_nonzero, x_nonzero = np.nonzero(left)
        # left = left[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        left = left[:np.max(y_nonzero) + 1, :]
        top = cv2.copyMakeBorder(top, 0, maxy - top.shape[0], 0, 0, cv2.BORDER_CONSTANT, 0)
        left = cv2.copyMakeBorder(left, 0, maxy - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, 0)

        cube = np.ones((leftx, topx, maxy))
        topT = np.transpose(top)
        leftT = np.transpose(left)

        for k in range(leftx):
            cube[k, :, :] = cube[k, :, :] * topT
        for k in range(topx):
            cube[:, k, :] = cube[:, k, :] * leftT
        for k in range(maxy):
            topk = int(sum(top[k]))
            leftk = int(sum(left[k]))
            if topk == 0 or leftk == 0:
                continue

            if usemiddle:
                middlev = getview(s, "middle")
                middlecircle = middlev.fills[i]
            else:
                middlecircle = cv2.circle(np.zeros((401, 401)), (200, 200), 200, 1, -1)
            y_nonzero, x_nonzero = np.nonzero(middlecircle)
            middlecircle = middlecircle[np.min(y_nonzero):np.max(y_nonzero) + 1,
                           np.min(x_nonzero):np.max(x_nonzero) + 1]
            middlers = cv2.resize(middlecircle, (int(sum(top[k])), int(sum(left[k]))), interpolation=cv2.INTER_NEAREST)
            y_nonzero, x_nonzero = np.nonzero(cube[:, :, k])
            x1 = min(x_nonzero)
            y1 = min(y_nonzero)
            x2 = max(x_nonzero)
            y2 = max(y_nonzero)
            xy1 = (min(x_nonzero), min(y_nonzero))
            xy2 = (max(x_nonzero), max(y_nonzero))
            l = np.sum(cube[:, :, k], axis=0)
            w = np.sum(cube[:, :, k], axis=1)

            try:
                cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers
            except:  # Not really sure when this would happen
                l2 = np.trim_zeros(l, 'fb') / max(l)
                w2 = np.trim_zeros(w, 'fb') / max(w)

                if len(l2) != sum(l2):  # The x y assignments here might be completely wrong
                    #print("length")
                    circletemplate = cv2.resize(middlecircle, (len(l2), len(w2)), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * circletemplate
                    zwhere = np.where(l2 == 0)[0]
                    dims = (zwhere[0], y2 - y1 + 1)
                    x1 = min(x_nonzero)
                    # x1 = min(lwhere)
                    x2 = x1 + zwhere[0] - 1
                    middlers = cv2.resize(middlecircle, (dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers
                    # dims = (y2 - y1 - zwhere[-1], x2 - x1 + 1)
                    x2 = max(x_nonzero)  # WHYYY IS THIS HERE AND NOT IN THE WIDTH ADDED 5-16-22
                    dims = (x2 - x1 - zwhere[-1], y2 - y1 + 1)  # FIXED
                    # dims = (x2 - x1 + 1, y2 - y1 + 1)  # FIXED MORE
                    x1 = x1 + zwhere[-1] + 1
                    x2 = max(x_nonzero)
                    middlers = cv2.resize(middlecircle, (dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers

                    x1 = min(x_nonzero)

                if len(w2) != sum(w2):
                    #print("width")
                    # display(cube[:,:,k])
                    # Right here is hacky but it works?
                    circletemplate = cv2.resize(middlecircle, (len(l2), len(w2)), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * circletemplate
                    zwhere = np.where(w2 == 0)[0]
                    dims = (x2 - x1 + 1, zwhere[0])
                    y1 = min(y_nonzero)
                    # y1 = min(wwhere)
                    y2 = y1 + zwhere[0] - 1
                    middlers = cv2.resize(middlecircle, (dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers
                    # display(cube[xy1[1]:xy2[1]+1, xy1[0]:xy2[0]+1, k]) # See the hackyness
                    y2 = max(y_nonzero)  # ADDED 5-16-22
                    dims = (x2 - x1 + 1, y2 - zwhere[-1] - y1)
                    y1 = min(y_nonzero) + zwhere[-1] + 1
                    y2 = max(y_nonzero)
                    middlers = cv2.resize(middlecircle, (dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
                    cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers

                # cube[y1:y2 + 1, x1:x2 + 1, k] = cube[y1:y2 + 1, x1:x2 + 1, k] * middlers

        # cube = ndimage.median_filter(cube, size=4)
        # cube = np.rot90(cube, 2, (0, 2))
        vol = round(sum(sum(sum(cube))) * 0.064 * 0.064 * 0.064, 3)
        s.vols.append(vol)
        s.cubes.append(cube.astype(np.uint8))

    s.cubes2 = []
    print("Hollowing...")
    for cube in s.cubes:
        cube2 = np.zeros(cube.shape, dtype=np.uint8)
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                row = cube[i, j, :]
                if np.all(row == 0):
                    continue
                nz = row.nonzero()[0]
                l = nz[0]
                r = nz[-1]
                cube2[i, j, l] += 1
                cube2[i, j, r] += 1
        for i in range(cube.shape[0]):
            for j in range(cube.shape[2]):
                row = cube[i, :, j]
                if np.all(row == 0):
                    continue
                nz = row.nonzero()[0]
                l = nz[0]
                r = nz[-1]
                cube2[i, l, j] += 1
                cube2[i, r, j] += 1
        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                row = cube[:, i, j]
                if np.all(row == 0):
                    continue
                nz = row.nonzero()[0]
                l = nz[0]
                r = nz[-1]
                cube2[l, i, j] += 1
                cube2[r, i, j] += 1
        cube2 = np.ceil(cube2 / np.max(cube2)).astype(np.uint8)
        cube2 = (cube * 2) - cube2
        s.cubes2.append(cube2.astype(np.uint8))

    del s.cubes


timer.check(True)

nsdetails(s)

writeinterm("3-volume", dd)


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