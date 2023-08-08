from dustpy import Simulation
from dustpylib.dynamics.backreaction import setup_backreaction
import pytest


def test_backreaction():
    s = Simulation()
    s.initialize()
    setup_backreaction(s)
    s.update


def test_backreaction_vertical():
    s = Simulation()
    s.initialize()
    setup_backreaction(s, vertical_setup=True)
    s.update
