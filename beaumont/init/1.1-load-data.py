# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *
runinit()
timer = runstartup()

vps = 0.5 # uL / sec
mmhg2kpa = 7.5006
samples = []

for pth, dirs, files in os.walk(rawdir):
    nonids = ["raw", "_data"]
    i = 0
    for i, f in enumerate([f for f in files if f[-3:].lower() == "csv"]):
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        fname = os.path.splitext(f)[0]
        id = fname[:-17].strip()
        id1 = os.path.split(pth)[1]
        id2 = os.path.split(os.path.split(pth)[0])[1]
        if id1 not in nonids:
            if is_date(id1[:10]):
                id1 = id1[10:].strip()
        else: id1 = None
        if id2 not in nonids:
            if is_date(id2[:10]):
                id2 = id2[10:].strip()
        else: id2 = None

        _label = fname.lower()
        print(letter, _label)

        csv = pd.read_csv(Path(pth, f), skiprows=1, names=["msec", "bits", "pressure"])
        pt = csv.drop(columns=["bits"])
        pt["p"] = pt["pressure"] - pt["pressure"].min()
        pt["pkpa"] = pt["p"]/mmhg2kpa
        pt["t"] = pt["msec"] - pt["msec"].iloc[0]
        if (pt["t"] < 0).any(): raise ValueError("Negative time")
        if pt["p"].max() < 25: print("Short pressure")

        s = ns(id=id, id1=id1, id2=id2, pth=pth, pt=pt, _label=_label, letter=letter)
        samples.append(s)

        # lowess = getlowess(pt["p"], pt["t"], 0.05)
        # plt.figure(dpi=300, figsize=(15, 6))
        # plt.scatter(pt["t"], pt["p"], marker=".")
        # plt.plot(lowess[:, 0], lowess[:, 1], marker=".", color="black")
        # plt.tight_layout()
        # plt.show()
        # plt.close()

samples = sorted(samples, key=lambda x: x._label)
dd = ns(const = ns(vps=vps, mmhg2kpa=mmhg2kpa), samples = samples)


#%%


for pth, dirs, files in os.walk(rawdir):
    for i, f in enumerate([f for f in files if f[-3:].lower() == "avi"]):
        _label = os.path.splitext(f)[0].lower()
        s = getsample(dd.samples, _label)
        s.vidfile = Path(pth, f)
        video = cv2.VideoCapture(str(s.vidfile))
        s.totalframes = int(video.get(7))
        s.width = int(video.get(3))
        s.height = int(video.get(4))
        video.release()
        offby = len(s.pt) - s.totalframes
        if 0 < offby < 5:
            s.pt = s.pt.iloc[:-offby]
        elif offby == 0: pass
        else: print("Mismatch", _label)


for s in dd.samples:
    while s.pt["t"].iloc[-1] < 5:  # Fix small times at end of data
        s.pt = s.pt[:-1]
        s.totalframes = s.totalframes - 1

    while s.pt["p"].iloc[-1] < (s.pt["p"].max() - 5):  # Remove small pressures
        s.pt = s.pt[:-1]
        s.totalframes = s.totalframes - 1

    filepath = vizdir / "pressure-time" / f"{_label}.png"
    #graphwide(pt["t"], pt["p"], "Time [sec]", "Pressure [mmHg]", title=_label, file=filepath)

    s.start = s.pt["p"].idxmin()
    s.intervals = np.around(np.linspace(s.start, s.totalframes-1, 100)).astype(int)


printkeys(s)

writeinterm("1-ptdata", dd)