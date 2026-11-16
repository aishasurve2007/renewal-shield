"""Shared test setup. Tests never call an API: the network is not needed and no key is read."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # the modules use paths relative to the project root (law/, results/)
