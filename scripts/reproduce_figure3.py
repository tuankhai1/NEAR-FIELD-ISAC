"""Convenience wrapper for ``python -m near_field_isac figure3``."""

import sys

from near_field_isac.cli import main

if __name__ == "__main__":
    main(["figure3", *sys.argv[1:]])

