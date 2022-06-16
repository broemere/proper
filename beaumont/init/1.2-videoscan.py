# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("1-ptdata")

bdir = outputdir / "brightness"
mkdirpy(bdir)
endsdir = intermdir / "ends"
mkdirpy(endsdir)
stacksize = 10


for s in dd.samples:  # 2 min per
    print("Scanning: " + s._label)
    s.frames = []
    errout = False

    while len(s.frames) < 100 and not errout:
        video = cv2.VideoCapture(str(s.vidfile))
        i = 0
        s.brightnesses = []
        s.frames = []
        framesums = []
        lastframe = None
        while i <= s.intervals[-1]:
            ret, frame = video.read()
            if not ret:
                videoreaderror(s, i)
                filename = s._label + "_error_" + str(i-1) + ".png"
                filepath = endsdir / filename
                cv2.imwrite(str(filepath), lastframe)
                print("Last before error saved")
                errout = True
                break
            fsum = np.sum(frame)
            if i == s.start:
                framesums.append(fsum)
            if i in s.intervals:
                if i == s.intervals[np.around(len(s.intervals)/2).astype(int)]:
                    print("\t50%")
                if fsum < (np.mean(framesums)/2):  # If frame is too dark then remake intervals and restart
                    print('\tDark frame', i, "- Restarting with new intervals")
                    darkidx = np.where(s.intervals == i)[0].item()
                    s.intervals = np.around(np.linspace(s.start, s.intervals[darkidx-1], 100)).astype(int)
                    break
                else:
                    s.frames.append(frame)
                    framesums.append(np.sum(frame))
                #s.frames.append(frame)

            if i == s.intervals[-1]:
                filename = s._label + "_stack.png"
                filepath = endsdir / filename
                canv = np.zeros(s.frames[0].shape)

                for j in s.frames:
                    canv += (j / 100)
                canv = canv/2
                canv += 0.25*s.frames[0]
                canv += 0.25*s.frames[-1]
                canv = np.floor(canv).astype(int)
                cv2.imwrite(str(filepath), canv)
                print("\tStack saved")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            s.brightnesses.append(brightness)
            i += 1
            lastframe = frame

        video.release()

    s.mb = np.median(s.brightnesses)

    s.p = s.pt["p"].iloc[s.intervals]
    s.t = s.pt["t"].iloc[s.intervals]

    plt.figure(dpi=250)
    plt.title(s._label + " Frame Brightness")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.plot(s.brightnesses)
    plt.ylim(ymin=0, ymax=(np.max(s.brightnesses)*1.25).astype(int))
    filename = s._label + "-brightness.png"
    plt.savefig(bdir / filename)
    plt.close()


printkeys(s)
getkeysizes(s)

writeinterm("1-videoscan", dd)
timer.check(True)
