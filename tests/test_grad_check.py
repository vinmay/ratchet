"""Test suite for ratchet.

Run at the start of every session:

    python -m pytest tests/ -q

Today it passes trivially. That is fine. The ritual needs to work before the
code does.

As you build, delete the `skip` marks below and make them pass. Each unit adds
tests here; nothing is considered done until its test is green.
"""

import pytest


def test_environment():
    """Smoke test. Confirms the harness runs and numpy is importable."""
    import numpy as np

    assert np.array([1.0, 2.0]).sum() == 3.0


# ---------------------------------------------------------------------------
# Unit 1 targets. Remove the skip marks as you implement them.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Unit 1: not implemented yet")
def test_forward_arithmetic():
    from micrograd.engine import Value

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    assert (a * b + c).data == 4.0


@pytest.mark.skip(reason="Unit 1: not implemented yet")
def test_reflected_operators():
    from micrograd.engine import Value

    x = Value(3.0)

    assert (2 * x).data == 6.0
    assert (2 + x).data == 5.0


@pytest.mark.skip(reason="Unit 1: not implemented yet")
def test_children_are_tracked():
    from micrograd.engine import Value

    a = Value(2.0)
    b = Value(3.0)
    c = a * b

    assert a in c._prev
    assert b in c._prev
    assert len(c._prev) == 2


@pytest.mark.skip(reason="Unit 1: not implemented yet")
def test_topo_sort_orders_parents_first():
    from micrograd.engine import Value, topo_sort

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a

    order = topo_sort(d)

    assert order.index(a) < order.index(c)
    assert order.index(c) < order.index(d)
    assert order[-1] is d


@pytest.mark.skip(reason="Unit 1: not implemented yet")
def test_shared_node_appears_once():
    """The diamond case. `x` is used twice but is one node in the graph.

    If this fails, `_prev` is a list, or topo_sort is not tracking visited
    nodes. Both bugs stay silent until Unit 2, then produce gradients that are
    exactly 2x too large.
    """
    from micrograd.engine import Value, topo_sort

    x = Value(3.0)
    y = x * x + x

    order = topo_sort(y)

    assert order.count(x) == 1
    assert y.data == 12.0


# ---------------------------------------------------------------------------
# Unit 2 starts here.
#
# `grad_check` is yours to write. It is the core instrument of this project
# and is on the do-not-delegate list in CLAUDE.md, so it is deliberately
# absent from this file.
#
# When you write it, it goes in this module and every primitive gets a test
# that compares its analytical gradient against the numerical one.
# ---------------------------------------------------------------------------