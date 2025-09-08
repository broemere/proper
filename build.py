#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys

# --- Configuration ---
APP_SCRIPT = 'proper.py'
CONFIG_SCRIPT = 'config.py'
SPEC_FILE = 'build.spec'
APP_BASE_NAME = 'proper'


def get_version():
    """Reads the APP_VERSION from the config.py script."""
    print("--- Reading version number from config.py ---")
    # This regex now looks for a variable named APP_VERSION
    version_re = r'^APP_VERSION\s*=\s*[\'"]([^\'"]+)[\'"]'

    # This now opens config.py instead of the main script
    with open(CONFIG_SCRIPT, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(version_re, line)
            if match:
                version = match.group(1)
                print(f"Version found: {version}\n")
                return version
    raise RuntimeError(f"Could not find APP_VERSION in {CONFIG_SCRIPT}.")


def clean():
    """Removes previous build artifacts."""
    print("--- Cleaning old build directories ---")
    for folder in ['build', 'dist', '__pycache__']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for f in os.listdir():
        if f.startswith(APP_BASE_NAME) and f.endswith(".zip"):
            os.remove(f)
    print("Clean complete.\n")


def build(version):
    """Runs the PyInstaller command after setting the version environment variable."""
    print("--- Running PyInstaller ---")

    # Set the version in an environment variable for the spec file to read (only for this process, not system)
    env = os.environ.copy()
    env['APP_VERSION'] = version

    command = ['pyinstaller', SPEC_FILE, '--clean']

    print(f"Executing: {' '.join(command)}")
    # Pass the modified environment to the subprocess
    subprocess.run(command, check=True, env=env)
    print("PyInstaller build successful!\n")


def archive(version):
    """Creates a distributable ZIP archive of the build."""
    print("--- Creating distributable archive ---")
    platform = 'mac' if sys.platform == 'darwin' else 'win'
    app_versioned_name = f"{APP_BASE_NAME}_v{version}"

    dist_name = app_versioned_name
    if platform == 'mac':
        dist_name += '.app'
    elif platform == 'win':
        dist_name += '.exe'

    archive_basename = f"{app_versioned_name}_{platform}"
    archive_name = os.path.join('dist', archive_basename)
    shutil.make_archive(archive_name, 'zip', root_dir='dist', base_dir=dist_name)
    print(f"Successfully created archive: {archive_name}.zip\n")


if __name__ == "__main__":
    #clean()
    app_version = get_version()
    build(app_version)
    archive(app_version)
    print("✅ Build process complete!")
