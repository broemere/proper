import certifi
import ssl
import logging
import urllib.request
from PySide6.QtCore import QThread, Signal
from config import APP_VERSION, REPO_URL


log = logging.getLogger(__name__)


class UpdateChecker(QThread):
    """
    Background thread to check for updates on GitHub.
    Emits 'update_available' signal with the new version string if a newer version is found.
    """
    update_available = Signal(str)

    def run(self):
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            # We use urllib to avoid adding an external dependency like 'requests'
            # The timeout ensures we don't hang indefinitely if the internet is flaky
            with urllib.request.urlopen(REPO_URL, context=ssl_context, timeout=5) as response:
                # GitHub redirects 'latest' to the specific tag URL.
                # Example: .../releases/tag/v0.1
                final_url = response.geturl()

                # Extract the tag (the last part of the URL)
                latest_tag = final_url.split('/')[-1]

                if self._is_version_newer(latest_tag, APP_VERSION):
                    self.update_available.emit(latest_tag)
        except Exception as e:
            # Log silently; we don't want to annoy the user if they are offline
            log.warning(f"Update check failed: {e}")

    def _is_version_newer(self, remote_tag: str, current_version: str) -> bool:
        """
        Compares two version strings (e.g., 'v0.1' vs '0.0.5').
        Returns True if remote_tag is logically greater than current_version.
        """
        try:
            # 1. Strip 'v' prefix and whitespace
            r_clean = remote_tag.lower().lstrip('v').strip()
            c_clean = current_version.lower().lstrip('v').strip()

            # 2. Convert to tuples of integers for accurate comparison
            # e.g. "0.1" -> (0, 1) and "0.0.5" -> (0, 0, 5)
            # Python natively compares tuples element-by-element:
            # (0, 1) > (0, 0, 5) evaluates to True because 1 > 0 at the second index.
            remote_parts = tuple(map(int, r_clean.split('.')))
            current_parts = tuple(map(int, c_clean.split('.')))

            return remote_parts > current_parts
        except ValueError:
            # Failsafe for tags that aren't standard numbers (e.g., "beta-release")
            log.warning(f"Could not parse version tags for comparison: {remote_tag} vs {current_version}")
            return False