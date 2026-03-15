from __future__ import annotations

from importlib import metadata


MIN_SEARCH_PACKAGE_VERSION = (0, 1, 3)


def _parse_version(version_text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version_text.split("."):
        digits = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            break
        parts.append(int("".join(digits)))
    return tuple(parts)


def import_run_search():
    try:
        from sas94_search_api.search_service import run_search
    except ImportError as exc:
        raise RuntimeError(
            "Missing search package dependency. Install `sas-94-search-api` "
            "before running chat/search features in this project."
        ) from exc
    try:
        installed_version = metadata.version("sas-94-search-api")
    except metadata.PackageNotFoundError:
        installed_version = None
    if installed_version is not None and _parse_version(installed_version) < MIN_SEARCH_PACKAGE_VERSION:
        raise RuntimeError(
            "Installed search package is too old. "
            "Install `sas-94-search-api>=0.1.3` before running chat features."
        )
    return run_search
