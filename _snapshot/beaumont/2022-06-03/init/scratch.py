# @author: broemere
# created: 6/2/2022
"""
~~~Description~~~
"""
from elib import *

#timer = runstartup()

ignoredirs = ["_data", "viz"]

cmpdirs = []
checkdirs = []
difffiles = []
difffiledirs = []
newfiles = []

scheduled = True
description = "manual"

snapshots = os.listdir(snapdir / projname)
mostrecent = datetime.date(2000, 1, 1)
for d in snapshots:
    if len(d) != 10:  # Ignore manual snapshots
        continue
    dtime = datetime.datetime.strptime(d, "%Y-%m-%d").date()
    if dtime > mostrecent:
        mostrecent = dtime

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
    difffiledirs += [sourcedir]*len(diffs)
    newfiles = newfiles + dircmp(sourcedir, checkdir).left_only



destname = str(datetime.date.today())
if not scheduled:
    destname = destname + " - " + description
if scheduled:
    delta = (datetime.date.today() - mostrecent).days
    if len(difffiles+newfiles) == 0 or delta < 7:
        # Don't snapshot if no files have been modified or it has been less than 1 week since
        #return None
        pass
    else:
        print("Saving snapshot...")
        destdir = snapdir / projname / destname
        #copytree(projdir, destdir, ignore=copytreeignoredirs)

for f, d in zip(difffiles, difffiledirs):
    diff = [item for item in d.parts if item not in projdir.parts]
    fname = os.path.splitext(f)[0] + ".diff"
    new = new = snapdir / projname / str(mostrecent)
    for part in diff:
        new = new.joinpath(part)
    p = Path(new, fname)
    #p.touch()




def copytreeignoredirs(dir, files):
    if dir == projdir:
        return ignoredirs
    return []



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
