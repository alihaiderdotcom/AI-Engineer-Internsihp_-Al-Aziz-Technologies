"""Make a small, timeout-aware request to a public API."""

import logging

import requests


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_python_release():
    """Return the latest Python release metadata from python.org."""
    url = "https://www.python.org/api/v2/downloads/release/?is_published=true"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        releases = response.json()
        if not releases:
            raise ValueError("The API returned no releases")
        return releases[0]
    except requests.RequestException as error:
        logger.error("API request failed: %s", error)
        return None
    except (ValueError, TypeError) as error:
        logger.error("API response was invalid: %s", error)
        return None


if __name__ == "__main__":
    release = fetch_python_release()
    if release is None:
        raise SystemExit(1)
    print(f"Latest release: {release.get('name', 'unknown')}")
