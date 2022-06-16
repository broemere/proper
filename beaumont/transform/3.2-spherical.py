# @author: broemere
# created: 6/12/2022
"""
~~~Description~~~
"""
from elib import *
timer = runstartup()
dd = loadinterm("3-shapes")

finalt = 0.08


for s in dd.samples:
    s.vols = np.array(s.vols)
    finalv = max(s.vols)
    finalr = ((finalv / math.pi) * (3 / 4)) ** (1 / 3)
    tissuevol = finalv - ((4 / 3) * math.pi * ((finalr - finalt) ** 3))

    rs = ((s.vols / math.pi) * (3 / 4)) ** (1 / 3)
    s.rsinner = (((s.vols - tissuevol) / math.pi) * (3 / 4)) ** (1 / 3)
    s.ts = rs - s.rsinner
    s.rplushalft = s.rsinner + (s.ts / 2)


for s in dd.samples:
    s.stress = (s.p * s.rplushalft) / (2 * s.ts)
    s.stretch = (s.rplushalft / s.rplushalft.min())


for s in dd.samples:
    s.spherevols = (4/3)*math.pi*(s.spherers**3)
    finalr = np.max(s.spherers)
    finalv = np.max(s.spherevols)
    tissuevol = finalv - ((4 / 3) * math.pi * ((finalr - finalt) ** 3))

    s.spherersinner = (((s.spherevols - tissuevol) / math.pi) * (3 / 4)) ** (1 / 3)
    s.spherets = s.spherers - s.spherersinner
    if np.any(s.spherets < 0):
        print(s.letter, "negative thickness!")
    s.sphererplushalft = s.spherersinner + (s.spherets / 2)


for s in dd.samples:
    s.spherestress = (s.p * s.sphererplushalft) / (2 * s.spherets)
    s.spherestretch = (s.sphererplushalft / np.delete(s.sphererplushalft, s.sphererplushalft.argmin()).min())


for s in dd.samples:
    s.ellipsevols = (s.da*s.db*s.dc)*(4/3)*math.pi

writeinterm("3-spherical", dd)

