# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("1-videoscan")

reddir = intermdir / "endsred"
mkdirpy(reddir)
upperthresh = 200  # Select color above this value
lowerthresh = 100  # Select color below this value
dd.const.padding = 25
dd.const.initial = True  # Set flag for initial red frames too
dd.const.middleview = True

kernel = np.ones((3,3),np.uint8)

#dd.samples = dd.samples[:2]

for s in dd.samples:
    print(s._label)
    filename = s._label + "_last_red.png"
    filepath = str(reddir / filename)
    segmentedframe = cv2.imread(filepath)
    redchannel = cv2.inRange(segmentedframe, (0, 0, upperthresh), (lowerthresh, lowerthresh, 255))
    greenchannel = cv2.inRange(segmentedframe, (0, upperthresh, 0), (lowerthresh, 255, lowerthresh))
    s.redsegments = nlargestcnts3(redchannel, dd.const.middleview+2)
    s.greensegments = nlargestcnts3(greenchannel, 5)
    s.redmask = np.zeros(redchannel.shape)
    for index, row in s.redsegments.iterrows():
        s.redmask = cv2.drawContours(s.redmask, [row["cnt"]], -1, 1, -1)
    s.redmask = s.redmask.astype(np.uint8)
    # canv = np.zeros(redchannel.shape)
    # for index, row in s.greensegments.iterrows():
    #     canv = cv2.drawContours(canv, [row["cnt"]], -1, 1, -1)

    redxs, redys = [], []
    for index, row in s.redsegments.iterrows():
        (x, y), r = cv2.minEnclosingCircle(row["cnt"])
        redxs.append(x)
        redys.append(y)
    greenxs, greenys = [], []
    for index, row in s.greensegments.iterrows():
        (x, y), r = cv2.minEnclosingCircle(row["cnt"])
        greenxs.append(x)
        greenys.append(y)
    bottomj = greenys.index(max(greenys))
    topj = greenys.index(min(greenys))
    leftj = greenxs.index(min(greenxs))
    rightj = greenxs.index(max(greenxs))
    centerj = list({0, 1, 2, 3, 4} - {topj, bottomj, leftj, rightj})[0]
    s.vieworder = {topj: "top", leftj: "left", centerj: "middle", rightj: "right", bottomj: "bottom"}
    s.vieworder = dict(sorted(s.vieworder.items()))
    redorder = []
    for j in range(len(redxs)):
        smallestdist = 100000
        closestk = np.nan
        rcenter = (redxs[j], redys[j])
        for k in range(len(greenxs)):
            gcenter = (greenxs[k], greenys[k])
            dist = math.sqrt(((gcenter[0] - rcenter[0]) ** 2) + ((gcenter[1] - rcenter[1]) ** 2))
            if dist < smallestdist:
                smallestdist = dist
                closestk = k
        redorder.append(closestk)
    redorder = pd.Series([0, 1, 2], index=redorder)
    s.views = []
    s.redviews = []

    for j in range(5):
        # print(s.vieworder[j])
        view = ns(location=s.vieworder[j], greencenter=(int(round(greenxs[j])), int(round(greenys[j]))))
        view.greencenter0 = view.greencenter
        if j in redorder.index:
            redi = redorder[j]
            redcnt = s.redsegments["cnt"].iloc[redi]
            view.redview = cv2.drawContours(np.zeros(redchannel.shape), [redcnt], -1, 1, -1)
            view.redcnt = redcnt
            s.redviews.append(s.vieworder[j])
        s.views.append(view)

    y_nonzero, x_nonzero = np.nonzero(s.redmask)
    s.topb3 = np.min(y_nonzero)
    s.bottomb3 = np.max(y_nonzero) + 1
    s.leftb3 = np.min(x_nonzero)
    s.rightb3 = np.max(x_nonzero) + 1
    # s.bottomb += 1
    # s.rightb += 1
    # s.cropbox = [s.topb, s.bottomb, s.leftb, s.rightb]
    s.cropbox3 = [s.topb3, s.bottomb3, s.leftb3, s.rightb3]
    # s.padbox = [(s.topb - dd.const.padding), (s.bottomb + dd.const.padding), (s.leftb - dd.const.padding),
    #             (s.rightb + dd.const.padding)]
    s.padbox3 = [(s.topb3 - dd.const.padding), (s.bottomb3 + dd.const.padding), (s.leftb3 - dd.const.padding),
                 (s.rightb3 + dd.const.padding)]
    s.redmask3 = s.redmask[s.padbox3[0]:s.padbox3[1], s.padbox3[2]:s.padbox3[3]].astype(np.uint8)
    # s.redmaskfull = deepcopy(s.redmask)
    # s.redmask = s.redmask[s.padbox[0]:s.padbox[1], s.padbox[2]:s.padbox[3]]
    s.rederode = cv2.erode(s.redmask3, kernel, iterations=1).astype(np.uint8)
    # display((s.redmask[s.topb:s.bottomb,s.leftb:s.rightb], True)

    frames = deepcopy(s.frames)
    s.frames = []
    for f in frames:  # Compress stored keyframes
        s.frames.append(f[s.padbox3[0]:s.padbox3[1], s.padbox3[2]:s.padbox3[3]])


nsdetails(dd.samples[0])

writeinterm("1-crop", dd)