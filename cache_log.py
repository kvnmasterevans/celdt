# cache_log.py

import os
import json

CACHE_LOG_PATH = "cache_log.json"


def load_cache_log():
    """Load the mapping between original filenames and PKL cache files."""
    if os.path.exists(CACHE_LOG_PATH):
        with open(CACHE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_cache_log(cache_log):
    """Persist the cache mapping."""
    with open(CACHE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_log, f, indent=4)


def register_cache(original_filename, cache_filename):
    """Register which PKL file belongs to an original input file."""
    cache_log = load_cache_log()

    cache_log[original_filename] = cache_filename

    save_cache_log(cache_log)


def get_cache_filename(original_filename):
    """Look up the PKL file associated with an original filename."""
    cache_log = load_cache_log()

    return cache_log.get(original_filename)