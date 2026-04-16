import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_forward import test_forward_pass
from test_losses import test_act_training_step

print("Running test_forward_pass...")
test_forward_pass()
print("Passed.")

print("Running test_act_training_step...")
test_act_training_step()
print("Passed.")

print("All tests passed.")
