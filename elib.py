# @author: broemere
# created: 1/12/2022
import matplotlib.pyplot as plt

from headers import *

reponame = "proper"

parentpath = Path(__file__).resolve().parents[1]
properpath = Path(__file__).resolve().parents[0]
propername = os.path.basename(properpath)
if propername != reponame:
    raise ValueError("elib.py is not in the project root dir!")

snapdir = properpath / "_snapshot"
execdir = Path().absolute()  # Get the folder script is executed in

projname, projdir = None, None

for i, d in enumerate(execdir.parents):
    if d.stem == reponame:
        projname = execdir.parts[-i - 1]
        projdir = Path(*execdir.parts[:-i])

if projname != None:
    datadir = projdir / "_data"
    rawdir = datadir / "raw"

    intermdir = datadir / "interm"
    outputdir = datadir / "output"
    vizdir = projdir / "viz"

ignoredirs = ["_data", "viz"]

#print(Path().absolute())


# =================================================== META TOOLS ===================================================== #

class starttimer:
    def __init__(self):
        self.t0 = perf_counter()
        self.tx = 0

    def check(self, minutes=False):
        """Return time since last check (or start if first check)
        Default seconds
        Pass True to return minutes
        """
        self.appendage = " sec"
        if self.tx == 0:
            self.tx = perf_counter()
            self.t_out = self.tx - self.t0
        else:
            self.tx2 = perf_counter()
            self.t_out = self.tx2 - self.tx
            self.tx = self.tx2
        if minutes:
            self.t_out = self.t_out / 60
            self.appendage = " min"
        #return str(int(self.t_out)) + self.appendage + "\n"
        print(str(int(self.t_out)) + self.appendage + "\n")

    def total(self, minutes=False):
        """Return time since start
        Default seconds
        Pass True for minutes
        """
        self.appendage = " sec"
        self.t_out = perf_counter() - self.t0
        if minutes:
            self.t_out = self.t_out / 60
            self.appendage = " min"
        #return str(int(self.t_out)) + self.appendage + "\n"
        print(str(int(self.t_out)) + self.appendage + "\n")


def snapshot_old(description="manual", scheduled=False):
    """(description, scheduled) Copy scripts to snapshot if there isn't one for this week already,
    and files have been changed"""
    print("...Checking snapshots...")
    if not os.path.exists(snapdir):
        raise ValueError("_snapshot folder does not exist in project parent dir!")
    if not os.path.exists(snapdir / projname):
        mkdirpy(snapdir / projname)

    snapshots = os.listdir(snapdir / projname)
    mostrecent = datetime.date(2000, 1, 1)
    for d in snapshots:
        if len(d) != 10:  # Ignore manual snapshots
            continue
        dtime = datetime.datetime.strptime(d, "%Y-%m-%d").date()
        if dtime > mostrecent:
            mostrecent = dtime

    sourcedirs = []
    filediffs = []

    if mostrecent == datetime.date(2000, 1, 1):
        filediffs = None
    else:
        for d in os.listdir(projdir):
            if d in ignoredirs:
                continue
            sourcepath = projdir / d
            sourcedirs.append(d)
            checkpath = snapdir / projname / str(mostrecent) / d
            filediffs.append(dircmp(sourcepath, checkpath).diff_files)

    # sourcepath = projdir / "script"
    #
    # else:
    #     checkpath = snapdir / projname / str(mostrecent)
    #     filediffs = dircmp(sourcepath, checkpath).diff_files
    destdir = str(datetime.date.today())

    if not scheduled:
        destdir = destdir + " - " + description
    if scheduled:
        delta = (datetime.date.today() - mostrecent).days
        if filediffs != None:
            flatfilediffs = [item for sublist in filediffs for item in sublist]
            if len(flatfilediffs) == 0 or delta < 7:
                # Only snapshot if files modified and last snapshot > 1 week old
                return None
    print("...Saving snapshot...")
    destpath = snapdir / projname / destdir
    #copytree(sourcepath, destpath)
    copytree(projdir, destpath, ignore=copytreeignoredirs)
    if filediffs != None:
        for flist, d in zip(filediffs, sourcedirs):  # Show changed files
            for f in flist:
                fname = os.path.splitext(f)[0] + ".diff"
                with open(destpath / d / fname, 'w') as fp:
                    pass

def snapshot(description="manual", scheduled=False):
    """(description, scheduled) Copy scripts to snapshot if there isn't one for this week already,
    and files have been changed"""
    print("...Checking snapshots...")

    if not os.path.exists(snapdir):
        raise ValueError("_snapshot folder does not exist in project parent dir!")
    if not os.path.exists(snapdir / projname):
        mkdirpy(snapdir / projname)

    snapshots = os.listdir(snapdir / projname)
    mostrecent = datetime.date(2000, 1, 1)
    for d in snapshots:
        if len(d) != 10:  # Ignore manual snapshots
            continue
        dtime = datetime.datetime.strptime(d, "%Y-%m-%d").date()
        if dtime > mostrecent:
            mostrecent = dtime

    cmpdirs = []
    checkdirs = []
    difffiles = []
    difffiledirs = []
    newfiles = []

    for path, dirs, files in os.walk(projdir):
        p = Path(path)
        if len(set.intersection(set(p.parts), set(ignoredirs))) == 0:
            if p != projdir:
                cmpdirs.append(p)
                diff = [item for item in p.parts if item not in projdir.parts]
                checkdir = snapdir / projname / str(mostrecent)
                for part in diff: checkdir = checkdir.joinpath(part)
                checkdirs.append(checkdir)

    for sourcedir, checkdir in zip(cmpdirs, checkdirs):
        if not checkdir.exists():
            newfiles += os.listdir(sourcedir)
            continue
        diffs = dircmp(sourcedir, checkdir).diff_files
        difffiles = difffiles + diffs
        difffiledirs += [sourcedir] * len(diffs)
        newfiles = newfiles + dircmp(sourcedir, checkdir).left_only

    destname = str(datetime.date.today())
    if not scheduled:
        destname = destname + " - " + description
    if scheduled:
        delta = (datetime.date.today() - mostrecent).days
        if len(difffiles + newfiles) == 0 or delta < 7:
            # Don't snapshot if no files have been modified or it has been less than 1 week since
            return None

    print("Saving snapshot...")
    destdir = snapdir / projname / destname
    copytree(projdir, destdir, ignore=copytreeignoredirs)

    for f, d in zip(difffiles, difffiledirs):
        diff = [item for item in d.parts if item not in projdir.parts]
        fname = os.path.splitext(f)[0] + ".diff"
        new = snapdir / projname / destname
        for part in diff:
            new = new.joinpath(part)
        p = Path(new, fname)
        p.touch()


def copytreeignoredirs(dir, files):
    if dir == projdir:
        return ignoredirs
    return []


def runstartup():
    snapshot(scheduled=True)
    return starttimer()
    # compareexcelsmulti(intermdir / "eliresults_new.xlsx", intermdir / "eliresults_new2.xlsx")
    # compareexcelsmulti2(intermdir / "naive-pressure-time.xlsx", intermdir / "naive-pressure-time2.xlsx")


def runinit():
    mkdirpy(datadir)
    mkdirpy(rawdir)
    mkdirpy(intermdir)
    mkdirpy(vizdir)
    mkdirpy(outputdir)
    mkdirpy(snapdir)


def beep():
    """Beep 3 times"""
    beeptime = 200
    beepfreq = 1000
    for i in range(3):
        winsound.Beep(beepfreq, beeptime)


def display(img, gray=False):
    fig = plt.figure(dpi=300, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if gray:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    plt.show()
    plt.close()


def graph(x, y, xlabel="", ylabel="", title="", file=""):
    plt.figure(dpi=300)
    plt.scatter(x, y, marker=".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if file != "":
        d = os.path.dirname(file)
        mkdirpy(d)
        plt.savefig(str(file))
        plt.close()
        #cv2.imwrite(str(file), plt)
    else:
        plt.show()
        plt.close()


def graphwide(x, y, xlabel="", ylabel="", title="", file=""):
    plt.figure(dpi=300, figsize=(15, 6))
    plt.scatter(x, y, marker=".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if file != "":
        d = os.path.dirname(file)
        f = os.path.basename(file)
        mkdirpy(d)
        plt.savefig(os.path.join(d, "wide_"+f))
        plt.close()
        #cv2.imwrite(str(file), plt)
    else:
        plt.show()
        plt.close()


# =================================================== FUNCTIONS ===================================================== #

def normalize(img):
    """Normalize gray img from 0 to 255 (uint8)
    Divide by 255 to get (float64)"""
    img = cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return img


def togray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to3channel(img):
    img = np.dstack((img, img, img))
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def rssq(v):
    return np.sqrt(np.sum(v ** 2))


def rmse(v):
    return np.sqrt(mean(v ** 2))


def is_date(string, fuzzy=False):
    try:
        parsedate(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False

def interpolation(d, x):
    """d data, middle part to interpolate"""
    output = d[1][0] + (x - d[0][0]) * ((d[1][1] - d[1][0]) / (d[0][1] - d[0][0]))
    return output

# =================================================== FILE HANDLING ================================================== #

def mkdirpy(path):
    """Creates directory if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def toexcel(data, filename, sheets, src, keepindex=False):
    """(data, filename, sheets, final=False, keepindex=False)
    Export data to .xlsx file. Pass pandas dataframe,
    or dict of data for sheets=True."""

    savepath = intermdir / src
    mkdirpy(savepath)
    excelfile = filename + ".xlsx"
    potentialrename = True
    dirlist = os.scandir(savepath)
    for f in dirlist:
        if f.is_file():
            filename2 = os.path.basename(f)
            if filename2 == excelfile:
                potentialrename = False

    if sheets:
        with pd.ExcelWriter(savepath / excelfile) as writer:
            for k, v in data.items():
                v.to_excel(writer, sheet_name=k, index=keepindex)
    else:
        with pd.ExcelWriter(savepath / excelfile) as writer:
            data.to_excel(writer, index=keepindex)
    # Check if the file is a rename
    if potentialrename:
        for f in os.scandir(savepath):
            if f.is_file():
                filename2 = os.path.basename(f)
                if filename2 != excelfile:
                    if os.path.splitext(filename2)[-1] == ".xlsx":
                        if compareexcels(savepath / excelfile, savepath / f):
                            print("Renaming file: " + filename2)
                            os.remove(savepath / filename2)


def writedata(path, data):
    """Dump data into pickle file in path"""
    with open(path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    return None


def loaddata(path):
    """Return data from pickle file in path"""
    with open(path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def writeinterm(filename, data, inparallel=False):
    """Dump data into pickle file in path"""
    filename += ".pick"
    filepath = intermdir
    if inparallel:
        filepath = intermdir / "_parallel-stores"
    mkdirpy(filepath)
    file = filepath / filename
    potentialrename = True
    #dirlist = os.scandir(intermdir)
    with open(file, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    # Check if there are identical files
    if potentialrename:
        for f in os.scandir(filepath):
            if f.is_file():
                filename2 = os.path.basename(f)
                if filename2 != filename:
                    if comparefiles(file, filepath / f):
                        print("Renaming file: " + filename2)
                        os.remove(filepath / f)
    print(f"-> {humanize.naturalsize(os.path.getsize(file))}")


def loadinterm(filename):
    """Return data from pickle file in path"""
    filename += ".pick"
    filepath = intermdir
    file = filepath / filename
    with open(file, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def videoreaderror(sample, i):
    print("Video length mismatch")
    print(sample._label)
    print("Current frame: " + str(i))
    print("Total frames: " + str(sample.totalframes))


def makevideo(filepath, data, fps):
    if len(data[0].shape) == 3:
        size = data[0].shape[::-1][1:3]
    else:
        size = data[0].shape[::-1]
    video = cv2.VideoWriter(str(filepath),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, size, 1)
    for img in data:
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

def outputvideo(filename, data, fps, subdir=""):
    filepath = outputdir / subdir
    mkdirpy(filepath)
    video = cv2.VideoWriter(str(filepath / filename),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, data[0].shape[0:2], 1)
    for img in data:
        video.write(img)
    cv2.destroyAllWindows()
    video.release()



# =================================================== GETTERS ======================================================== #

def getlowess(x, y, f):
    return sm.nonparametric.lowess(x, y, frac=f)

def getsheetdetails(file_path):
    # https://stackoverflow.com/questions/12250024
    sheets = []
    file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    dir_name = os.path.dirname(file_path)
    # Make a temporary directory with the file name
    directory_to_extract_to = os.path.join(dir_name, file_name)
    mkdirpy(directory_to_extract_to)
    # Extract the xlsx file as it is just a zip file
    zip_ref = ZipFile(file_path, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()
    # Open the workbook.xml which is very light and only has meta data, get sheets from it
    path_to_workbook = os.path.join(directory_to_extract_to, 'xl', 'workbook.xml')
    with open(path_to_workbook, 'r') as f:
        xml = f.read()
        dictionary = xmltodict.parse(xml)
        if not isinstance(dictionary['workbook']['sheets']['sheet'], list):
            sheets.append(dictionary['workbook']['sheets']['sheet']['@name'])
        else:
            for sheet in dictionary['workbook']['sheets']['sheet']:
                sheets.append(sheet['@name'])
    # Delete the extracted files directory
    f.close()
    rmtree(directory_to_extract_to)
    return sheets


def getsample(samples, label):
    #if label[0] != "0":
    #    if label.find("0") > 0:
    #        label = label[:2].zfill(3) + label[2:]
    #    else:
    #        label = label[0].zfill(3) + label[1:]
    for s in samples:
        if s._label.lower() == label.lower():
            return s
    print("Could not find " + label)
    return None


def printkeys(s, br=False):
    if br:
        print("{")
        for k in sorted(s.__dict__.keys()):
            print("\t", k)
        print("}")
    else:
        print(sorted(s.__dict__.keys()))

def getkeysizes(s):  # Return size of keys if large enough to matter
    tempdir = intermdir / "temp"
    mkdirpy(tempdir)
    for k, v in s.__dict__.items():
        filename = k + ".pick"
        file = tempdir / filename
        with open(file, 'wb') as pickle_file:
            pickle.dump(v, pickle_file)
        fsize = humanize.naturalsize(os.path.getsize(file))
        if fsize[-1] == "B" and fsize[-2:] != "kB":
            print(f"{k}: {fsize}")
    rmtree(tempdir)


def getkeysizesall(s):  # Return size of keys if large enough to matter
    tempdir = intermdir / "temp"
    mkdirpy(tempdir)
    for k, v in s.__dict__.items():
        filename = k + ".pick"
        file = tempdir / filename
        with open(file, 'wb') as pickle_file:
            pickle.dump(v, pickle_file)
        fsize = humanize.naturalsize(os.path.getsize(file))
        print(f"{k}: {fsize}")
    rmtree(tempdir)


def nsdetails(s, br=False):
    print("----------------- SAMPLE METADATA ----------------")
    printkeys(s, br)
    getkeysizes(s)


def getview(sample, view):
    """Takes sample, and view string title e.g. 'top'
    Returns corresponding view"""
    for v in sample.views:
        if v.location == view:
            return v
    print("View not found")
    return None


def getframes(sample, ddframes):
    for s in ddframes.samples:
        if s._label == sample._label:
            return s.frames
    print("Frames not found")
    return None

def getattrs(dd, attr):

    for s in dd.samples:
        pass

# =================================================== ALGORITHMS ===================================================== #

def nlargestcnts1(img, n, simple=True):
    """(img, n, simple=True) Find n largest contours in binary image.
    CCOMP method, ignores holes. Pass 0 for CHAIN_APPROX_NONE.
    Returns sorted dict {area: contour}. Smallest to largest."""
    if simple:
        approx_meth = cv2.CHAIN_APPROX_SIMPLE
    else:
        approx_meth = cv2.CHAIN_APPROX_NONE
    cnts, hiers = cv2.findContours(img, cv2.RETR_CCOMP, approx_meth)
    bign = {}
    for i in range(len(cnts)):
        if hiers[0][i][3] >= 0:  ## Ignore holes
            continue
        i = cnts[i]
        a = cv2.contourArea(i)
        if len(bign) < n:
            bign[a] = i
        else:
            smallest = min(bign)
            if a > smallest:
                del bign[smallest]
                bign[a] = i
    bign = dict(sorted(bign.items()))
    return bign


def nlargestcnts2(img, n, simple=True):
    """(img, n, simple=True) Find n largest contours in binary image.
    TREE method. Pass 0 for CHAIN_APPROX_NONE.
    Returns sorted dict {area: contour}. Smallest to largest."""
    if simple:
        approx_meth = cv2.CHAIN_APPROX_SIMPLE
    else:
        approx_meth = cv2.CHAIN_APPROX_NONE
    cnts, hiers = cv2.findContours(img, cv2.RETR_TREE, approx_meth)
    bign = {}
    for i in cnts:
        a = cv2.contourArea(i)
        if len(bign) < n:
            bign[a] = i
        else:
            smallest = min(bign)
            if a > smallest:
                del bign[smallest]
                bign[a] = i
    bign = dict(sorted(bign.items()))
    return bign


def nlargestcnts3(img, n, simple=True):
    """(img, n, simple=True) Find n largest contours in binary image.
    TREE method. Pass 0 for CHAIN_APPROX_NONE.
    Deals with 0 and very small area contours
    Returns a dataframe of [area, cnt]"""
    if simple:
        approx_meth = cv2.CHAIN_APPROX_SIMPLE
    else:
        approx_meth = cv2.CHAIN_APPROX_NONE
    cnts, hiers = cv2.findContours(img, cv2.RETR_TREE, approx_meth)
    areas = np.array([])
    cntarray = []
    for j in cnts:
        a = cv2.contourArea(j)
        if len(areas) < n:
            areas = np.append(areas, a)
            cntarray.append(j)
        else:
            smlst = np.argmin(areas)
            if a > areas[smlst] or areas[smlst] == 0:
                areas = np.delete(areas, smlst)
                del cntarray[smlst]
                areas = np.append(areas, a)
                cntarray.append(j)

    bign = pd.DataFrame()
    bign["area"] = areas
    bign["cnt"] = cntarray
    return bign


def ls_ellipsoid(xx, yy, zz):
    # http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))
    K = np.ones_like(x)  # column of ones

    # np.hstack performs a loop over all samples and creates
    # a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC = np.dot(InvJTJ, np.dot(JT, K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC, -1)

    return (eansa)


def polyToParams3D(vec, printMe):
    # http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)

    # Algebraic form: X.T * Amat * X --> polynomial form

    if printMe: print("\npolynomial\n" + str(vec))

    Amat = np.array(
        [
            [vec[0], vec[3] / 2.0, vec[4] / 2.0, vec[6] / 2.0],
            [vec[3] / 2.0, vec[1], vec[5] / 2.0, vec[7] / 2.0],
            [vec[4] / 2.0, vec[5] / 2.0, vec[2], vec[8] / 2.0],
            [vec[6] / 2.0, vec[7] / 2.0, vec[8] / 2.0, vec[9]]
        ])

    if printMe: print("\nAlgebraic form of polynomial\n" + str(Amat))

    # See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3, 0:3]
    A3inv = inv(A3)
    ofs = vec[6:9] / 2.0
    center = -np.dot(A3inv, ofs)
    if printMe: print("\nCenter at:" + str(center))

    # Center the ellipsoid at the origin
    Tofs = np.eye(4)
    Tofs[3, 0:3] = center
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))
    if printMe: print("\nAlgebraic form translated to center\n" + str(R) + "\n")

    R3 = R[0:3, 0:3]
    R3test = R3 / R3[0, 0]
    if printMe: print("normed \n" + str(R3test))
    s1 = -R[3, 3]
    R3S = R3 / s1
    (el, ec) = eig(R3S)

    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)
    if printMe: print("\nAxes are\n" + str(axes) + "\n")

    inve = inv(ec)  # inverse is actually the transpose here
    if printMe: print("\nRotation matrix\n" + str(inve))
    return (center, axes, inve)


def printAns3D(center, axes, R, xin, yin, zin, verbose):
    # http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    print("\nCenter at  %10.4f,%10.4f,%10.4f" % (center[0], center[1], center[2]))
    print("Axes gains %10.4f,%10.4f,%10.4f " % (axes[0], axes[1], axes[2]))
    print("Rotation Matrix\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f" % (
        R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2]))

    # Check solution
    # Convert to unit sphere centered at origin
    #  1) Subtract off center
    #  2) Rotate points so bulges are aligned with axes (no xy,xz,yz terms)
    #  3) Scale the points by the inverse of the axes gains
    #  4) Back rotate
    # Rotations and gains are collected into single transformation matrix M

    # subtract the offset so ellipsoid is centered at origin
    xc = xin - center[0]
    yc = yin - center[1]
    zc = zin - center[2]

    # create transformation matrix
    L = np.diag([1 / axes[0], 1 / axes[1], 1 / axes[2]])
    M = np.dot(R.T, np.dot(L, R))
    print("\nTransformation Matrix\n" + str(M))

    # apply the transformation matrix
    [xm, ym, zm] = np.dot(M, [xc, yc, zc])
    # Calculate distance from origin for each point (ideal = 1.0)
    rm = np.sqrt(xm * xm + ym * ym + zm * zm)

    print("\nAverage Radius  %10.4f (truth is 1.0)" % (np.mean(rm)))
    print("Stdev of Radius %10.4f\n " % (np.std(rm)))

    return


def ls_sphere(xx, yy, zz):
    # http: // www.juddzone.com / ALGORITHMS / least_squares_sphere.html
    asize = np.size(xx)
    # print("Sphere input size is " + str(asize))
    J = np.zeros((asize, 4))
    ABC = np.zeros(asize)
    K = np.zeros(asize)

    for ix in range(asize):
        x = xx[ix]
        y = yy[ix]
        z = zz[ix]

        J[ix, 0] = x * x + y * y + z * z
        J[ix, 1] = x
        J[ix, 2] = y
        J[ix, 3] = z
        K[ix] = 1.0

    K = K.transpose()  # not required here
    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ)

    ABC = np.dot(InvJTJ, np.dot(JT, K))
    # If A is negative, R will be negative
    A = ABC[0]
    B = ABC[1]
    C = ABC[2]
    D = ABC[3]

    xofs = -B / (2 * A)
    yofs = -C / (2 * A)
    zofs = -D / (2 * A)
    R = np.sqrt(4 * A + B * B + C * C + D * D) / (2 * A)
    if R < 0.0: R = -R
    return (xofs, yofs, zofs, R)


def connect(ends):
    # https://stackoverflow.com/questions/47704008/fastest-way-to-get-all-the-points-between-two-x-y-coordinates-in-python
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1:
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0 + 1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], d0 + 1))
                         .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1 + 1))
                         .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], d1 + 1, dtype=np.int32)]


@jit
def connect2(ends):
    # https://stackoverflow.com/questions/47704008/fastest-way-to-get-all-the-points-between-two-x-y-coordinates-in-python
    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1):
        return np.c_[np.arange(ends[0, 0], ends[1, 0] + np.sign(d0), np.sign(d0), dtype=np.int32),
                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0) // 2,
                               ends[0, 1] * np.abs(d0) + np.abs(d0) // 2 + (np.abs(d0) + 1) * d1, d1,
                               dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1) // 2,
                               ends[0, 0] * np.abs(d1) + np.abs(d1) // 2 + (np.abs(d1) + 1) * d0, d0,
                               dtype=np.int32) // np.abs(d1),
                     np.arange(ends[0, 1], ends[1, 1] + np.sign(d1), np.sign(d1), dtype=np.int32)]


@jit
def connect_nd(ends):
    # https://stackoverflow.com/questions/47704008/fastest-way-to-get-all-the-points-between-two-x-y-coordinates-in-python
    d = np.diff(ends.T).T[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)
    return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD


@jit
def boundary_check(cntpts, origin, r, hullorigin):
    n = 90
    dist = np.sqrt(np.abs(origin[1] - hullorigin[1]) ** 2 + np.abs(origin[0] - hullorigin[0]) ** 2)
    r2 = r + dist + 2
    linestack = [0] * n
    cntd = {}
    for i, xy in enumerate(cntpts):
        cntd[(xy[0], xy[1])] = i

    for k in range(n):
        #m = int(k*(360/n))
        a = math.radians(k * (360 / n))
        y = int(round(r2 * math.sin(a))) + origin[0]
        x = int(round(r2 * math.cos(a))) + origin[1]
        #y = int(round(r2 * sinlist[m])) + origin[0]
        #x = int(round(r2 * coslist[m])) + origin[1]
        intmpts = connect_nd(np.array([[origin[1], origin[0]], [x, y]]))
        for pt in intmpts:
            if (pt[0], pt[1]) in cntd:
                linestack[k] = 1
                break
    return linestack


def cannyframe(frame, sample, close=False, kernel=np.ones((3,3),np.uint8)):
    frame = normalize((togray(frame).astype(np.int16()) * (sample.mb / np.mean(togray(frame)))))
    fmedian = cv2.medianBlur(frame, 15, 15) * sample.redmask3
    cny = canny(normalize(fmedian))
    cny = cny * sample.rederode
    if close:
        cny = cv2.morphologyEx(cny, cv2.MORPH_CLOSE, kernel)
    return cny

def ocf1frame(cny, sample, smoothsize=11, kernel=np.ones((3,3),np.uint8), kernelo=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))):
    for v in sample.redviews:
        view = getview(sample, v)
        if view.location in sample.redviews:
            view.fills = []
            view.gc = (view.greencenter[0] - sample.padbox3[2], view.greencenter[1] - sample.padbox3[0])
            view.lastrpaint = 2
            view.lastrscan = 75

    for v in sample.redviews:
        print(v)
        view = getview(sample, v)
        cnymask = cny * view.redview[sample.padbox3[0]:sample.padbox3[1], sample.padbox3[2]:sample.padbox3[3]]
        center = (int(round(view.gc[0])), int(round(view.gc[1])))
        floodfill = cv2.circle(np.zeros(cnymask.shape), center, view.lastrpaint, 1, -1)

        y_nonzero, x_nonzero = np.nonzero(view.redview[sample.padbox3[0]:sample.padbox3[1], sample.padbox3[2]:sample.padbox3[3]])
        view.redbox = [np.min(y_nonzero) - 2, np.max(y_nonzero) + 3, np.min(x_nonzero) - 2, np.max(x_nonzero) + 3]
        cnymask = cnymask[view.redbox[0]:view.redbox[1], view.redbox[2]:view.redbox[3]]
        cntpts = np.transpose(np.where(cnymask == 1))

        floodfill = floodfill[view.redbox[0]:view.redbox[1], view.redbox[2]:view.redbox[3]]
        floodfillold = deepcopy(floodfill)
        blacklist = []

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

                linestack = boundary_check(cntpts, origin, view.lastrscan * 1.25, center)
                linestack = np.array(linestack)
                if np.count_nonzero(linestack) < 60:
                    blacklist.append(origin)
                    floodfill[origin[1], origin[0]] = 0
            if np.max(sumfill) < 2:  # Quick check if we are still inside all boundaries
                continue
            if np.array_equal(floodfill, floodfillold):
                # print("ay")
                break
            floodfillold = deepcopy(floodfill)

        canv = floodfill + cnymask
        canv = cv2.erode(canv, kernelo, iterations=1)
        canv = cv2.morphologyEx(canv, cv2.MORPH_OPEN, kernel)
        canvcnts = nlargestcnts3(canv.astype(np.uint8), 1)
        cnts, hiers = cv2.findContours(canv.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        canv = np.zeros(canv.shape)
        for index, row in canvcnts.iterrows():
            canv = cv2.drawContours(canv, [row["cnt"]], -1, 1, -1)
        canv = cv2.morphologyEx(canv, cv2.MORPH_CLOSE, kernel)
        canv = cv2.medianBlur(canv.astype(np.uint8), smoothsize)
        view.fills.append(canv.astype(np.uint8))

    return sample


# =================================================== COMPARES ===================================================== #

def comparefiles(file1, file2):
    "Do the two files have exactly the same contents?"
    # https://stackoverflow.com/a/255210
    with open(file1, "rb") as fp1, open(file2, "rb") as fp2:
        if os.fstat(fp1.fileno()).st_size != os.fstat(fp2.fileno()).st_size:
            return False  # different sizes ∴ not equal
        # set up one 4k-reader for each file
        fp1_reader = functools.partial(fp1.read, 4096)
        fp2_reader = functools.partial(fp2.read, 4096)
        # pair each 4k-chunk from the two readers while they do not return '' (EOF)
        cmp_pairs = zip(iter(fp1_reader, b''), iter(fp2_reader, b''))
        # return True for all pairs that are not equal
        inequalities = itertools.starmap(operator.ne, cmp_pairs)
        # voilà; any() stops at first True value
        return not any(inequalities)


def compareexcels(file1, file2):
    file1sheets = getsheetdetails(file1)
    file2sheets = getsheetdetails(file2)
    if file1sheets != file2sheets:
        return False
    with open(file1, "rb") as fp1, open(file2, "rb") as fp2:
        size1 = os.fstat(fp1.fileno()).st_size
        size2 = os.fstat(fp2.fileno()).st_size
        if abs(size1 - size2) > 3:
            return False  # different sizes (with some wiggle room) ∴ not equal
    fp1.close()
    fp2.close()
    df1 = load_workbook(file1, read_only=True)
    df2 = load_workbook(file2, read_only=True)
    print("Running full excel compare...")
    for sheet in file1sheets:
        sheet1 = df1[sheet]
        sheet2 = df2[sheet]
        if sheet1.calculate_dimension() != sheet2.calculate_dimension():
            return False
        sentinel = object()
        for a, b in itertools.zip_longest(sheet1.values, sheet2.values, fillvalue=sentinel):
            if a != b:
                return False
    df1.close()
    df2.close()
    return True


def comparesheetsmulti(sheetparts):
    # df1 = sheetparts[0]
    # df2 = sheetparts[1]
    # for sheet in sheetparts[2]:
    #     sheet1 = pd.read_excel(df1, sheet_name=sheet)
    #     sheet2 = pd.read_excel(df2, sheet_name=sheet)
    #     print("ticking")

    for comps in sheetparts:
        for a, b in comp:
            if a != b:
                return False
    return True


def grabsheetcompare(idx, df1, df2):
    sheet1 = df1.worksheets[idx]
    sheet2 = df2.worksheets[idx]
    sentinel = object()
    for a, b in itertools.zip_longest(sheet1.values, sheet2.values, fillvalue=sentinel):
        if a != b:
            return False
    return True


def comparesheetsmulti2(fq, file1, file2, outq):
    df1 = load_workbook(file1, read_only=True, data_only=True)
    df2 = load_workbook(file2, read_only=True, data_only=True)

    truthtable = []
    while True:
        try:
            idx = fq.get_nowait()
        except:
            # fq.put(truthtable)
            outq.put(truthtable)
            exit(0)
        truthtable.append(grabsheetcompare(idx, df1, df2))

    # df1 = sheetparts[0]
    # df2 = sheetparts[1]
    # for sheet in sheetparts[2]:
    #     sheet1 = pd.read_excel(df1, sheet_name=sheet)
    #     sheet2 = pd.read_excel(df2, sheet_name=sheet)
    #     print("ticking")

    # for comps in sheetparts:
    #     for a, b in comp:
    #         if a != b:
    #             return False
    # return True


def compareexcelsmulti(file1, file2):
    timer = starttimer()
    file1sheets = getsheetdetails(file1)
    file2sheets = getsheetdetails(file2)
    if file1sheets != file2sheets:
        return False
    with open(file1, "rb") as fp1, open(file2, "rb") as fp2:
        size1 = os.fstat(fp1.fileno()).st_size
        size2 = os.fstat(fp2.fileno()).st_size
        if abs(size1 - size2) > 3:
            return False  # different sizes (with some wiggle room) ∴ not equal
    fp1.close()
    fp2.close()

    ncores = multiprocessing.cpu_count() - 1
    npartitions = ncores
    pool = multiprocessing.Pool(ncores)

    df1 = load_workbook(file1, read_only=True, data_only=True)
    df2 = load_workbook(file2, read_only=True, data_only=True)
    # for sheet in file1sheets:
    #     sheet1 = df1[sheet]
    #     sheet2 = df2[sheet]
    #     if sheet1.calculate_dimension() != sheet2.calculate_dimension():
    #         return False

    print("Running full excel compare...")

    nsheets = len(file1sheets)
    poolsize = math.ceil(nsheets / ncores)
    print(poolsize)
    print(ncores)

    sheetpool = []
    isheet = 0

    # with pd.ExcelFile(file1) as df1, pd.ExcelFile(file2) as df2:

    # df11 = pd.read_excel(df1, sheet_name=None)
    # df22 = pd.read_excel(df2, sheet_name=None)
    # for i in range(ncores):
    #     sheetpool.append([df1, df2, []])
    #     #sheetpool.append([])
    #     for j in range(poolsize):
    #         sheetpool[i][2].append([])
    #         sheetpool[i][2][j].append(file1sheets[isheet])
    #         isheet += 1
    #         if isheet == nsheets:
    #             break

    for i in range(ncores):
        sheetpool.append([])
        for j in range(poolsize):
            sheetpool[i].append([])
            sentinel = object()
            sheetcomp = itertools.zip_longest(df1[file1sheets[isheet]], df2[file1sheets[isheet]], fillvalue=sentinel)
            sheetpool[i][j].append(sheetcomp)
            isheet += 1
            if isheet == nsheets:
                break

    print(sheetpool[0])

    truthtable = pool.map(comparesheetsmulti, sheetpool)
    pool.close()
    pool.join()

    print(truthtable)

    #     sentinel = object()
    #     for a, b in itertools.zip_longest(sheet1.values, sheet2.values, fillvalue=sentinel):
    #         if a != b:
    #             return False
    df1.close()
    df2.close()
    print(timer.total())
    return True


def compareexcelsmulti2(file1, file2):
    timer = starttimer()
    file1sheets = getsheetdetails(file1)
    file2sheets = getsheetdetails(file2)
    if file1sheets != file2sheets:
        return False
    with open(file1, "rb") as fp1, open(file2, "rb") as fp2:
        size1 = os.fstat(fp1.fileno()).st_size
        size2 = os.fstat(fp2.fileno()).st_size
        if abs(size1 - size2) > 3:
            return False  # different sizes (with some wiggle room) ∴ not equal
    fp1.close()
    fp2.close()

    ncores = multiprocessing.cpu_count() - 1
    npartitions = ncores
    pool = multiprocessing.Pool(ncores)

    df1 = load_workbook(file1, read_only=True, data_only=True)
    df2 = load_workbook(file2, read_only=True, data_only=True)
    # for sheet in file1sheets:
    #     sheet1 = df1[sheet]
    #     sheet2 = df2[sheet]
    #     if sheet1.calculate_dimension() != sheet2.calculate_dimension():
    #         return False

    print("Running full excel compare...")

    nsheets = len(file1sheets)
    poolsize = math.ceil(nsheets / ncores)
    print(poolsize)
    print(ncores)

    sheetpool = []
    isheet = 0
    fq = multiprocessing.Queue()
    outq = multiprocessing.Queue()
    for i in range(nsheets):
        fq.put(i)

    tt = []
    pool = [multiprocessing.Process(target=comparesheetsmulti2, args=(fq, file1, file2, outq)) for p in range(ncores)]

    for p in pool:
        p.start()

    for p in pool:
        p.join()

    for i in range(ncores):
        print(outq.get())
    # with pd.ExcelFile(file1) as df1, pd.ExcelFile(file2) as df2:

    # df11 = pd.read_excel(df1, sheet_name=None)
    # df22 = pd.read_excel(df2, sheet_name=None)
    # for i in range(ncores):
    #     sheetpool.append([df1, df2, []])
    #     #sheetpool.append([])
    #     for j in range(poolsize):
    #         sheetpool[i][2].append([])
    #         sheetpool[i][2][j].append(file1sheets[isheet])
    #         isheet += 1
    #         if isheet == nsheets:
    #             break

    # for i in range(ncores):
    #     sheetpool.append([])
    #     for j in range(poolsize):
    #         sheetpool[i].append([])
    #         sentinel = object()
    #         sheetcomp = itertools.zip_longest(df1[file1sheets[isheet]], df2[file1sheets[isheet]], fillvalue=sentinel)
    #         sheetpool[i][j].append(sheetcomp)
    #         isheet += 1
    #         if isheet == nsheets:
    #             break

    # print(sheetpool[0])

    # truthtable = pool.map(comparesheetsmulti, sheetpool)
    # pool.close()
    # pool.join()

    # print(truthtable)

    #     sentinel = object()
    #     for a, b in itertools.zip_longest(sheet1.values, sheet2.values, fillvalue=sentinel):
    #         if a != b:
    #             return False
    df1.close()
    df2.close()
    print(timer.total())
    return True


###

def carve(s):
    print(s.letter)
    usemiddle = True
    s.vols = []
    s.cubes = []
    for i in range(len(s.intervals)):
        if i == int(len(s.intervals) / 2):
            print("\t50%")
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
                    # print("length")
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
                    # print("width")
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
    s.vols = np.array(s.vols)
    s.cubes = np.array(s.cubes)

    return s


def hollow(s):
    print("\t", "Hollowing...")
    s.cubes2 = []
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
    s.cubes2 = np.array(s.cubes2)

    return s





def main():
    #exitcode = input("exit")
    pass

if __name__ == "__main__":
    main()