# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("1-crop")

cannydir = outputdir / "canny"
mkdirpy(cannydir)
savefigs = True

#%%
kernel = np.ones((3,3),np.uint8)

for s in dd.samples:  # 10 sec per
    print(s._label)
    rederode = cv2.erode(s.redmask3, kernel, iterations = 1)
    strip = np.ones((s.padbox3[1] - s.padbox3[0], 2))

    s.cannys = []
    for i in range(len(s.intervals)):
        catra = np.ndarray(s.redmask3.shape)
        frame = s.frames[i]
        frame = normalize((togray(frame).astype(np.int16()) * (s.mb / np.mean(togray(frame)))))
        fmedian = cv2.medianBlur(frame, 15, 15) * s.redmask3
        cny = canny(normalize(fmedian))
        cny = (cny * rederode).astype(np.uint8)
        # display(cny, True)
        s.cannys.append(cny)

        if savefigs and i % 10 == 0:
            strip = np.concatenate((strip, cny),axis=1)
            strip = np.concatenate((strip, np.ones((s.padbox3[1] - s.padbox3[0], 2))), axis=1)
    if savefigs:
        filename = s.letter + "-canny.png"
        filepath = cannydir / filename
        cv2.imwrite(str(filepath), strip*255)

    #del s.frames

nsdetails(s)

writeinterm("1-canny", dd)