from __future__ import annotations


def import_run_search():
    try:
        from sas94_search_api.search_service import run_search
    except ImportError as exc:
        raise RuntimeError(
            "Missing search package dependency. Install `sas-94-search-api` "
            "before running chat/search features in this project."
        ) from exc
    return run_search
