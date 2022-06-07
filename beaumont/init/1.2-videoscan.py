# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("1-ptdata")

bdir = intermdir / "brightness"
mkdirpy(bdir)
endsdir = intermdir / "ends"
mkdirpy(endsdir)


for s in dd.samples:
    print("Scanning: " + s._label)
    video = cv2.VideoCapture(str(s.vidfile))

    i = 0
    s.brightnesses = []
    s.frames = []
    lastframe = None
    while i < s.totalframes:
        ret, frame = video.read()
        if not ret:
            videoreaderror(s, i)
            filename = s._label + "_last_" + str(i-1) + ".png"
            filepath = endsdir / filename
            cv2.imwrite(str(filepath), lastframe)
            print("Last saved")
            break
        if i == s.start:
            filename = s._label + "_first.png"
            filepath = endsdir / filename
            cv2.imwrite(str(filepath), frame)
            print("First saved")
        if i == s.totalframes-1:
            filename = s._label + "_last.png"
            filepath = endsdir / filename
            cv2.imwrite(str(filepath), frame)
            print("Last saved")
        if i in s.intervals:
            s.frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        s.brightnesses.append(brightness)
        i += 1
        lastframe = frame

    video.release()
    s.mb = np.median(s.brightnesses)

    plt.figure(dpi=250)
    plt.title(s._label + " Frame Brightness")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.plot(s.brightnesses)
    filename = s._label + "-brightness.png"
    plt.savefig(bdir / filename)
    plt.close()


printkeys(s)
getkeysizes(s)

writeinterm("1-videoscan", dd)
timer.check(True)
