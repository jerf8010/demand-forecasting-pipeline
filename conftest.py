"""Make the `model/` package and its flat sibling modules importable from the tests."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "model"), os.path.join(ROOT, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
