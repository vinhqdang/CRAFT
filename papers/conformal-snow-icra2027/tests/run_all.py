import os
import sys

import pytest

if __name__ == "__main__":
    exit_code = pytest.main([os.path.dirname(os.path.abspath(__file__)), "-v"])
    sys.exit(exit_code)
