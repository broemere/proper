# @author: broemere
# created: 6/3/2022
"""
~~~Description~~~
"""
import sys
sys.path.insert(1, "D:\\PhD\\proper")
from elib import *

dd = loadinterm("1-canny")

magicnumber = 52
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
smoothsize = 5 # Must be odd


#%%

work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])

def ocffirst(s):
    for v in s.redviews:
        view = getview(s, v)
        if view.location in s.redviews:
            view.fills = []
            view.greencenter = (view.greencenter[0] - s.padbox3[2], view.greencenter[1] - s.padbox3[0])
            view.lastrpaint = 2
            view.lastrscan = 75

    # for cny, frameloc in zip(s.cannys, s.intervals):
    for cny, frameloc in zip(s.cannys, s.intervals):
        # print(str(s.intervals.index(frameloc)+1) + "/" + str(8) + " - " + str(frameloc))
        print(str(s.intervals.index(frameloc) + 1) + "/" + str(len(s.intervals)) + " - " + str(frameloc))
        for v in s.redviews:
            view = getview(s, v)
            cnymask = cny * view.redview[s.padbox3[0]:s.padbox3[1], s.padbox3[2]:s.padbox3[3]]
            # center = (int(round(view.greencenter[0] - s.padbox3[2])), int(round(view.greencenter[1] - s.padbox3[0])))
            center = (int(round(view.greencenter[0])), int(round(view.greencenter[1])))
            floodfill = cv2.circle(np.zeros(cnymask.shape), center, view.lastrpaint, 1, -1)
            fillorigin = center

            y_nonzero, x_nonzero = np.nonzero(view.redview[s.padbox3[0]:s.padbox3[1], s.padbox3[2]:s.padbox3[3]])
            view.redbox = [np.min(y_nonzero) - 2, np.max(y_nonzero) + 3, np.min(x_nonzero) - 2, np.max(x_nonzero) + 3]
            cnymask = cnymask[view.redbox[0]:view.redbox[1], view.redbox[2]:view.redbox[3]]
            cntpts = np.transpose(np.where(cnymask == 1))

            floodfill = floodfill[view.redbox[0]:view.redbox[1], view.redbox[2]:view.redbox[3]]
            floodfillold = deepcopy(floodfill)
            blacklist = []
            print("view", s.letter)

            while True:
                floodfill = cv2.dilate(floodfill, kernel, iterations=1)
                sumfill = cnymask + floodfill
                floodfill[sumfill == 2] = 0
                newpix = np.where((floodfill - floodfillold) > 0)  # Returns tuple of x and y

                for i in range(len(newpix[0])):
                    x = newpix[0][i]
                    y = newpix[1][i]
                    origin = (y, x)
                    if origin in blacklist:
                        floodfill[origin[1], origin[0]] = 0
                        continue

                    linestack = boundary_check(cntpts, origin, view.lastrscan * 1.25, fillorigin)
                    linestack = np.array(linestack)
                    if np.count_nonzero(linestack) < magicnumber:
                        blacklist.append(origin)
                        floodfill[origin[1], origin[0]] = 0
                if np.max(sumfill) < 2:  # Quick check if we are still inside all boundaries
                    continue
                if np.array_equal(floodfill, floodfillold):
                    # print("ay")
                    break
                # if np.sum(floodfill) > 4800:
                #     print("Area break")
                #     break

                floodfillold = deepcopy(floodfill)

            canv = floodfill + cnymask
            cnts, hiers = cv2.findContours(canv.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            canv = np.zeros(canv.shape)
            for cnt in cnts:
                canv = cv2.drawContours(canv, [cnt], -1, 1, -1)

            # canv = cv2.erode(canv, kernel, iterations=1)
            canv = cv2.morphologyEx(canv, cv2.MORPH_CLOSE, kernel)
            canv = cv2.medianBlur(canv.astype(np.uint8), smoothsize)
            view.fills.append(canv.astype(np.uint8))

            cnts, hiers = cv2.findContours(canv.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(cnts) > 1:
                print("Extra contours")
                cnts = [nlargestcnts3(canv.astype(np.uint8), 1, 0)['cnt'].item()]
                view.fills.pop()
                canv = np.zeros(canv.shape)
                for cnt in cnts:
                    canv = cv2.drawContours(canv, [cnt], -1, 1, -1)
                view.fills.append(canv.astype(np.uint8))

            for cnt in cnts:
                rectxy, widthheight, rot = cv2.minAreaRect(cnt)
                denom = 1.75 + (np.log10(1 + 175 * (s.totalframes - frameloc) / (s.totalframes)))
                view.lastrpaint = int(np.min(widthheight) / denom)
                (x, y), r = cv2.minEnclosingCircle(cnt)
                view.lastrscan = r
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00']) + view.redbox[2]
                cy = int(M['m01'] / M['m00']) + view.redbox[0]
                view.greencenter = (cx, cy)

        #break




def pool_handler3():
    p = mp.Pool(2)
    p.map(ocffirst, dd.samples)


if __name__ == '__main__':
    timer = runstartup()

    for s in dd.samples:
        s.intervals = s.intervals.tolist()
        #ocffirst(s)
    timer.check()

    pool_handler3()
    timer.check()

    for s in dd.samples:
        ocffirst(s)
    timer.check()


    exitcode = input("Exit")

