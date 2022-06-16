# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
import sys
sys.path.insert(1, "/")
sys.path.insert(1, "../..\\")
from elib import *

dd = loadinterm("1-canny")

magicnumber = 52
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
smoothsize = 5 # Must be odd

poolsize = 10

def updatelabel(s):

    print(s._label)
    s.lbl = s.letter + "_" + s._label
    return s


if __name__ == '__main__':
    timer = runstartup()

    for s in dd.samples:
        s.intervals = s.intervals.tolist()
        #ocffirst(s)
    timer.check()

    p = mp.Pool(poolsize)
    samples = p.map(updatelabel, dd.samples)



    timer.check()

    for s in samples:
        print(s.lbl)

    #for s in dd.samples:
    #    ocffirst(s)
    #timer.check()

    #writeinterm("2-ocf-parallel", dd)

    exitcode = input("Exit")