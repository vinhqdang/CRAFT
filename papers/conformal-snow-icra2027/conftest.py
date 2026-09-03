import os
import sys

# Make the repo root (holding craf_x/) importable regardless of the cwd
# pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
