"""Convenience wrapper for ``python -m near_field_isac figure4``."""

import sys

from near_field_isac.cli import main

if __name__ == "__main__":
    main(["figure4", *sys.argv[1:]])

