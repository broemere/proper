import os
from errno import EEXIST
from pathlib import Path, PurePath
import logging
import sys
import psutil
import subprocess

logger = logging.getLogger(__name__)


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


def resolve_cross_platform_path(stored_file_path: str) -> Path | None:
    """
    Intelligently converts a cross-platform file path to the current OS by
    finding the overlapping parent directory structure.
    """
    if not stored_file_path:
        return None

    # 1. Normalize slashes and isolate the parent directory
    normalized_str = stored_file_path.replace('\\', '/')
    target_dir = PurePath(normalized_str).parent

    # 2. Break the directory path into individual components
    parts = list(target_dir.parts)

    # Strip off common root anchors ('/', 'T:\', 'C:')
    if parts and (parts[0] == '/' or ':' in parts[0]):
        parts.pop(0)

    # Fetch available network drives using your existing function
    drives = list_network_drives()
    if not drives:
        logger.warning("No network drives detected on this system.")
        return None

    # 3. The Sliding Window Search (Directory Mode)
    for i in range(len(parts)):
        # Create a relative path from the current slice of parts
        relative_suffix = Path(*parts[i:])

        # Prevent false positives by avoiding checking an empty suffix at the drive root
        if str(relative_suffix) == ".":
            continue

        for drive in drives:
            mountpoint = Path(drive["mountpoint"])
            candidate_path = mountpoint / relative_suffix

            try:
                # Targeted OS-level check for the directory
                if candidate_path.is_dir():
                    logger.info(f"Resolved parent directory: {candidate_path}")
                    return candidate_path
            except OSError as e:
                # Catch permission errors, offline drives, etc., and move on
                logger.debug(f"Skipping {mountpoint} due to access error: {e}")
                continue

    logger.debug(f"Could not resolve parent directory on current network drives: {stored_file_path}")
    return None


def list_network_drives():
    """
    Return a list of dicts for each partition that looks like a network volume.
    On Windows, we detect network drives by checking part.device.startswith("\\\\").
    On macOS/Linux, we detect via fstype ∈ NETWORK_FS_TYPES or "remote" ∈ opts.
    """
    drives = []
    try:
        for part in psutil.disk_partitions(all=True):
            # ==== Windows: UNC‐style device path (e.g. '\\\\SERVER\\Share') ====
            if sys.platform.startswith("win") and part.device.startswith(r"\\"):
                if part.device.startswith(r"\\") or 'remot' in part.opts:
                    drives.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "opts": part.opts,
                    })

            # ==== macOS/Linux (and also catches any other network fs that psutil knows) ====
            else:
                ft = part.fstype.upper()
                is_network_fstype = ft in {"CIFS", "SMBFS","NFS", "AFPFS", "WEBDAV", "DAVFS"}
                is_remote_flag = "remote" in part.opts.lower()
                if is_network_fstype or is_remote_flag:
                    drives.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "opts": part.opts,
                    })
    except Exception as e:
        logger.error(f"Error scanning system partitions: {e}")

    # Add Windows Network Locations (Shortcuts)
    if sys.platform.startswith("win"):
        drives.extend(get_windows_network_locations())

    # Remove duplicates (in case a location is both mapped AND has a shortcut)
    # We key by the 'mountpoint' (path)
    unique_drives = {d['mountpoint']: d for d in drives}.values()

    return list(unique_drives)


def get_windows_network_locations():
    """
    Scans the Windows 'Network Shortcuts' folder for 'Add Network Location' items.
    Returns a list of dicts similar to psutil structure.
    """
    locations = []
    if not sys.platform.startswith("win"):
        return locations

    # The standard location where Windows stores "Network Locations"
    shortcut_dir = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Network Shortcuts"

    if not shortcut_dir.exists():
        return locations

    for item in shortcut_dir.iterdir():
        unc_path = None

        # Case A: The item is a direct .lnk file (e.g., "MyShare.lnk")
        if item.suffix.lower() == ".lnk":
            unc_path = resolve_shortcut_target(item)

        # Case B: The item is a folder containing a 'target.lnk' (Common for WebDAV/older wizards)
        elif item.is_dir():
            target_lnk = item / "target.lnk"
            if target_lnk.exists():
                unc_path = resolve_shortcut_target(target_lnk)

        if unc_path:
            locations.append({
                "device": unc_path,  # The actual \\Server\Share path
                "mountpoint": unc_path,  # Treated as the mountpoint for search
                "fstype": "NetworkLocation",
                "opts": "rw",  # Assumed
                "origin": "shortcut"  # Just a tag to know where this came from
            })

    return locations

def resolve_shortcut_target(shortcut_path):
    """
    Uses PowerShell to resolve the target of a Windows .lnk file.
    This avoids needing the heavy 'pywin32' library.
    """
    try:
        # PowerShell command to load WScript.Shell and read the target path
        cmd = [
            "powershell", "-NoProfile", "-Command",
            f"(New-Object -ComObject WScript.Shell).CreateShortcut('{str(shortcut_path)}').TargetPath"
        ]
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        target = result.stdout.strip()

        # Only return if it looks like a network path (UNC)
        if target.startswith(r"\\"):
            return target
    except Exception as e:
        logger.debug(f"Could not resolve shortcut {shortcut_path}: {e}")
    return None

home = make_dir("data")
