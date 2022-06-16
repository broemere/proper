# @author: broemere
# created: 6/3/2022
"""
Alternate OCF/voxellization script for processing in parallel. Run in terminal.
"""
import sys
sys.path.insert(1, "..\\..\\")
from elib import *
dd = loadinterm("2-ocf-parallel")


poolsize = 10


def carveandhollow(s):
    s = hollow(carve(s))
    del s.cubes
    return s


if __name__ == '__main__':
    timer = runstartup()
    p = mp.Pool(poolsize)
    dd.samples = p.map(carveandhollow, dd.samples)
    timer.check(True)

    print([s.letter for s in dd.samples])
    nsdetails(dd.samples[-1])
    writeinterm("2-vox-parallel", dd)
    exitcode = input("Exit")

