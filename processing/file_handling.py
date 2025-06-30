import os
from errno import EEXIST


def make_dir(path):
    """Creates directory if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return path


def file_exists(filepath):
    directory, filename = os.path.split(filepath)
    # If no directory is provided, assume current directory
    directory = directory or "."
    try:
        for f in os.listdir(directory):
            if f.lower() == filename.lower():
                return True
    except FileNotFoundError:
        # The directory doesn't exist
        return False
    return False

home = make_dir("data")
