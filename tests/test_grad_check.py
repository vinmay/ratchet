"""Test suite for ratchet.

Run at the start of every session:

    python -m pytest tests/ -q

"""

import math

import pytest
from micrograd.engine import Value, topo_sort, grad_check

def test_environment():
    """Smoke test. Confirms the harness runs and numpy is importable."""
    import numpy as np

    assert np.array([1.0, 2.0]).sum() == 3.0


# ---------------------------------------------------------------------------
# Unit 1
# ---------------------------------------------------------------------------


def test_forward_arithmetic():

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    assert (a * b + c).data == 4.0


def test_reflected_operators():
    from micrograd.engine import Value

    x = Value(3.0)

    assert (2 * x).data == 6.0
    assert (2 + x).data == 5.0


def test_children_are_tracked():

    a = Value(2.0)
    b = Value(3.0)
    c = a * b

    assert a in c._prev
    assert b in c._prev
    assert len(c._prev) == 2

def test_repr_is_a_readable_string():

    v = Value(4.0)
    s = repr(v)

    assert isinstance(s, str)
    assert "4.0" in s
    assert "Value" in s
    assert "object at 0x" not in s          # not the default repr

def test_repr_does_not_recurse_into_parents():
    """A repr that prints _prev blows the stack on any real graph."""

    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = a * b + c

    repr(out._prev)                          # would RecursionError if it walked parents
    assert len(repr(out)) < 60               # one node's worth, not the graph

def test_topo_sort_orders_parents_first():

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a

    order = topo_sort(d)

    assert order.index(a) < order.index(c)
    assert order.index(c) < order.index(d)
    assert order[-1] is d

def test_shared_node_appears_once():
    """The diamond case. `x` is used twice but is one node in the graph.

    If this fails, `_prev` is a list, or topo_sort is not tracking visited
    nodes. Both bugs stay silent until Unit 2, then produce gradients that are
    exactly 2x too large.
    """
    

    x = Value(3.0)
    y = x * x + x

    order = topo_sort(y)

    assert order.count(x) == 1
    assert y.data == 12.0


# ---------------------------------------------------------------------------
# Unit 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f, x, expected", [
    (lambda x: x + 4.0,  2.0,  1.0),      # constant slope
    (lambda x: 3.0 * x,  2.0,  3.0),      # scaled
    (lambda x: x * x,    3.0,  6.0),      # 2x, the same variable twice
    (lambda x: x ** 3,   2.0, 12.0),      # 3x^2
    (lambda x: 1.0 / x,  2.0, -0.25),     # -1/x^2, and negative
    (math.sin,           0.0,  1.0),      # cos(0)
    (math.exp,           1.0,  math.e),   # its own derivative
])
def test_grad_check_matches_known_derivatives(f, x, expected):
    assert grad_check(f, x) == pytest.approx(expected, rel=1e-6)


def test_grad_check_h_is_overridable():
    assert grad_check(lambda x: x * x, 3.0, h=1e-3) == pytest.approx(6.0, rel=1e-5)


def test_grad_check_does_not_evaluate_at_the_point_itself():
    """A one-point formula cannot measure a slope. Catches `f(x) / (2 * h)`."""
    calls = []

    def f(x):
        calls.append(x)
        return x * x

    grad_check(f, 3.0)

    assert len(calls) == 2
    assert 3.0 not in calls           # both evaluations are offset from x


# ---------------------------------------------------------------------------
# `__add__` backward. These seed `.grad` by hand and call `_backward()`
# directly, because `backward()` does not exist yet. Once it does, it should
# make the seeding lines unnecessary, not make these tests wrong.
# ---------------------------------------------------------------------------


def test_forward_pass_leaves_grads_at_zero():
    a, b = Value(2.0), Value(-3.0)
    c = a + b

    assert a.grad == 0.0
    assert b.grad == 0.0
    assert c.grad == 0.0


def test_leaf_backward_is_a_no_op():
    """Every node needs a callable `_backward`, or backward() dies on leaves."""
    a = Value(2.0)
    a.grad = 1.0

    a._backward()

    assert a.grad == 1.0


def test_add_routes_gradient_to_both_parents():
    a, b = Value(2.0), Value(-3.0)
    c = a + b
    c.grad = 1.0

    c._backward()

    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_add_scales_by_the_incoming_gradient():
    a, b = Value(2.0), Value(-3.0)
    c = a + b
    c.grad = 3.0

    c._backward()

    assert a.grad == pytest.approx(3.0)
    assert b.grad == pytest.approx(3.0)


def test_add_gradient_ignores_operand_values():
    """d(a+b)/da is 1 regardless of what a and b hold."""
    a, b = Value(1e6), Value(-500.0)
    c = a + b
    c.grad = 1.0

    c._backward()

    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_add_accumulates_when_a_node_is_used_twice():
    """`x + x` is one node reached twice. `=` instead of `+=` gives 1.0 here."""
    x = Value(3.0)
    y = x + x
    y.grad = 1.0

    y._backward()

    assert x.grad == pytest.approx(2.0)


def test_add_accumulates_and_scales_together():
    x = Value(1.0)
    y = x + x
    y.grad = 3.0

    y._backward()

    assert x.grad == pytest.approx(6.0)


def test_add_backward_survives_a_raw_number_operand():
    z = Value(5.0)
    w = z + 2
    w.grad = 1.0

    w._backward()

    assert z.grad == pytest.approx(1.0)


def test_radd_backward_survives_a_raw_number_operand():
    q = Value(4.0)
    r = 2 + q
    r.grad = 1.0

    r._backward()

    assert q.grad == pytest.approx(1.0)


def test_add_matches_grad_check():
    """The two independent computations of the same number must agree."""
    a = Value(2.0)
    c = a + 3.0
    c.grad = 1.0
    c._backward()

    assert a.grad == pytest.approx(grad_check(lambda x: x + 3.0, 2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# `backward()`: seed the root, walk the topo order in reverse.
# ---------------------------------------------------------------------------


def test_backward_seeds_the_root_gradient():
    a, b = Value(2.0), Value(-3.0)
    c = a + b

    c.backward()

    assert c.grad == pytest.approx(1.0)


def test_backward_reaches_both_parents():
    a, b = Value(2.0), Value(-3.0)
    c = a + b

    c.backward()

    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_backward_on_a_leaf_only_seeds_it():
    a = Value(2.0)

    a.backward()

    assert a.grad == pytest.approx(1.0)


def test_backward_accumulates_across_a_chain():
    """Three uses of x across two + nodes."""
    x = Value(3.0)
    y = x + x + x

    y.backward()

    assert x.grad == pytest.approx(3.0)


def test_backward_reaches_a_deep_leaf():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = a + b + c

    out.backward()

    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)
    assert c.grad == pytest.approx(1.0)


def test_backward_does_not_zero_grads_first():
    """Documents the convention: accumulation is the caller's to reset.

    Unit 3's training loop needs an explicit zero_grad between steps. If this
    test starts failing, backward() grew a zeroing step and the loop should
    drop its own.
    """
    x = Value(3.0)
    y = x + x

    y.backward()
    assert x.grad == pytest.approx(2.0)

    y.backward()
    assert x.grad == pytest.approx(4.0)


def test_backward_matches_grad_check_on_a_sum():
    a = Value(2.0)
    out = a + 3.0 + 4.0

    out.backward()

    assert a.grad == pytest.approx(grad_check(lambda x: x + 3.0 + 4.0, 2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# `__mul__` backward. The local derivative is the *other* operand, which is
# what separates multiplication from addition.
# ---------------------------------------------------------------------------


def test_mul_sends_each_parent_the_other_operand():
    """Swapping the two .data reads passes x*x but fails here."""
    a, b = Value(2.0), Value(-3.0)
    c = a * b

    c.backward()

    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)


def test_mul_accumulates_when_both_operands_are_the_same_node():
    """d(x*x)/dx = 2x. Suppressing the second contribution gives x."""
    x = Value(3.0)
    y = x * x

    y.backward()

    assert x.grad == pytest.approx(6.0)


def test_mul_with_equal_but_distinct_nodes():
    """Same value, different objects. Identity, not equality, is what matters."""
    u, w = Value(2.0), Value(2.0)
    z = u * w

    z.backward()

    assert u.grad == pytest.approx(2.0)
    assert w.grad == pytest.approx(2.0)


def test_mul_scales_by_the_incoming_gradient():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.grad = 4.0

    c._backward()

    assert a.grad == pytest.approx(-12.0)
    assert b.grad == pytest.approx(8.0)


def test_mul_by_a_raw_number_from_either_side():
    p = Value(2.0)
    (p * 5).backward()
    assert p.grad == pytest.approx(5.0)

    r = Value(2.0)
    (5 * r).backward()
    assert r.grad == pytest.approx(5.0)


@pytest.mark.parametrize("build, f, x0", [
    (lambda v: v * Value(-3.0), lambda t: t * -3.0,     2.0),
    (lambda v: v * v,           lambda t: t * t,        3.0),
    (lambda v: v * 5,           lambda t: t * 5,        2.0),
    (lambda v: 5 * v,           lambda t: 5 * t,        2.0),
    (lambda v: v * v + v,       lambda t: t * t + t,    3.0),
    (lambda v: (v + 2) * v,     lambda t: (t + 2) * t,  4.0),
    (lambda v: v * v * v,       lambda t: t * t * t,    2.0),
    (lambda v: (v + v) * (v * v), lambda t: (t + t) * (t * t), 1.5),
])
def test_backward_matches_grad_check(build, f, x0):
    """Analytic and numerical gradients, computed independently, must agree."""
    v = Value(x0)
    out = build(v)

    out.backward()

    assert v.grad == pytest.approx(grad_check(f, x0), rel=1e-6)


# ---------------------------------------------------------------------------
# `__pow__`. One parent, not two: the exponent is a constant, not a node.
# ---------------------------------------------------------------------------


def test_pow_forward():
    assert (Value(3.0) ** 2).data == pytest.approx(9.0)
    assert (Value(2.0) ** -1).data == pytest.approx(0.5)
    assert (Value(4.0) ** 0.5).data == pytest.approx(2.0)


def test_pow_has_a_single_parent():
    """The exponent is not differentiated, so it must not enter the graph."""
    x = Value(3.0)
    y = x ** 2

    assert y._prev == {x}


def test_pow_rejects_a_value_exponent():
    with pytest.raises(TypeError):
        Value(2.0) ** Value(3.0)


@pytest.mark.parametrize("k, x0", [
    (2, 3.0),        # the case where k*x accidentally equals k*x**(k-1)
    (3, 2.0),        # the case that separates them
    (1, 4.0),
    (0, 5.0),        # derivative of a constant
    (-1, 2.0),       # negative exponent
    (0.5, 4.0),      # fractional exponent
    (2, -3.0),       # negative base
])
def test_pow_backward_matches_grad_check(k, x0):
    v = Value(x0)
    out = v ** k

    out.backward()

    assert v.grad == pytest.approx(grad_check(lambda t: t ** k, x0), rel=1e-6)


def test_pow_accumulates_into_a_reused_node():
    x = Value(3.0)
    y = x ** 2 + x

    y.backward()

    assert x.grad == pytest.approx(grad_check(lambda t: t ** 2 + t, 3.0), rel=1e-6)


def test_pow_chains_through_other_operations():
    z = Value(2.0)
    w = (z * z) ** 3

    w.backward()

    assert z.grad == pytest.approx(grad_check(lambda t: (t * t) ** 3, 2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# `exp`. Its own derivative, so the closure reads the output value.
# ---------------------------------------------------------------------------


def test_exp_forward():
    assert Value(0.0).exp().data == pytest.approx(1.0)
    assert Value(1.0).exp().data == pytest.approx(math.e)
    assert Value(-2.0).exp().data == pytest.approx(math.exp(-2.0))


def test_exp_has_a_single_parent():
    x = Value(1.0)
    assert x.exp()._prev == {x}


@pytest.mark.parametrize("x0", [0.0, 1.0, -2.0, 0.5])
def test_exp_backward_matches_grad_check(x0):
    v = Value(x0)
    out = v.exp()

    out.backward()

    assert v.grad == pytest.approx(grad_check(math.exp, x0), rel=1e-6)


def test_exp_accumulates_when_reached_twice():
    """`=` instead of `+=` halves this. Single-use tests cannot see it."""
    z = Value(1.0)
    w = z.exp() + z.exp()

    w.backward()

    assert z.grad == pytest.approx(2 * math.e, rel=1e-6)


def test_exp_chains_through_other_operations():
    x = Value(2.0)
    y = (x * x).exp()

    y.backward()

    assert x.grad == pytest.approx(grad_check(lambda t: math.exp(t * t), 2.0), rel=1e-5)


def test_exp_nested_in_itself():
    q = Value(0.5)
    r = (q.exp() * q).exp()

    r.backward()

    expected = grad_check(lambda t: math.exp(math.exp(t) * t), 0.5)
    assert q.grad == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# `tanh`. Derivative is 1 - tanh(x)**2, expressible from the output value.
# ---------------------------------------------------------------------------


def test_tanh_forward():
    assert Value(0.0).tanh().data == pytest.approx(0.0)
    assert Value(1.0).tanh().data == pytest.approx(math.tanh(1.0))
    assert Value(-1.0).tanh().data == pytest.approx(-math.tanh(1.0))


def test_tanh_has_a_single_parent():
    x = Value(1.0)
    assert x.tanh()._prev == {x}


def test_tanh_is_bounded():
    """Saturates towards +/-1. Unit 4 will care about this."""
    assert Value(20.0).tanh().data == pytest.approx(1.0)
    assert Value(-20.0).tanh().data == pytest.approx(-1.0)


@pytest.mark.parametrize("x0", [0.0, 0.5, -1.0, 2.0, 5.0])
def test_tanh_backward_matches_grad_check(x0):
    v = Value(x0)
    out = v.tanh()

    out.backward()

    assert v.grad == pytest.approx(grad_check(math.tanh, x0), rel=1e-6)


def test_tanh_gradient_is_one_at_zero():
    v = Value(0.0)
    v.tanh().backward()

    assert v.grad == pytest.approx(1.0)


def test_tanh_gradient_vanishes_when_saturated():
    """The dead-gradient signature Unit 4 asks you to reproduce."""
    v = Value(5.0)
    v.tanh().backward()

    assert v.grad == pytest.approx(1 - math.tanh(5.0) ** 2, rel=1e-6)
    assert v.grad < 1e-3


def test_tanh_accumulates_when_reached_twice():
    z = Value(0.7)
    w = z.tanh() + z.tanh()

    w.backward()

    assert z.grad == pytest.approx(grad_check(lambda t: 2 * math.tanh(t), 0.7), rel=1e-6)


def test_tanh_chains_through_other_operations():
    x = Value(1.5)
    y = (x * x).tanh()

    y.backward()

    assert x.grad == pytest.approx(grad_check(lambda t: math.tanh(t * t), 1.5), rel=1e-5)


# ---------------------------------------------------------------------------
# `relu`. Piecewise, so the derivative is piecewise. Not differentiable at 0:
# the slope is 0 from the left and 1 from the right, and grad_check straddles
# the kink and returns 0.5. These tests assert the chosen convention instead.
# ---------------------------------------------------------------------------


def test_relu_forward():
    assert Value(3.0).relu().data == pytest.approx(3.0)
    assert Value(-2.0).relu().data == pytest.approx(0.0)
    assert Value(0.0).relu().data == pytest.approx(0.0)


def test_relu_forward_returns_floats_on_both_branches():
    assert isinstance(Value(-2.0).relu().data, float)


def test_relu_has_a_single_parent():
    x = Value(1.0)
    assert x.relu()._prev == {x}


@pytest.mark.parametrize("x0", [3.0, 0.5, -2.0, -0.1])
def test_relu_backward_matches_grad_check_away_from_zero(x0):
    v = Value(x0)
    out = v.relu()

    out.backward()

    assert v.grad == pytest.approx(
        grad_check(lambda t: t if t > 0 else 0.0, x0), abs=1e-6
    )


def test_relu_gradient_at_zero_follows_the_chosen_convention():
    """Not differentiable here. This pins the choice; grad_check would say 0.5."""
    v = Value(0.0)
    v.relu().backward()

    assert v.grad == pytest.approx(0.0)


def test_relu_kills_gradient_on_the_negative_branch():
    """The dead-ReLU signature Unit 4 asks you to reproduce."""
    v = Value(-2.0)
    v.relu().backward()

    assert v.grad == pytest.approx(0.0)


def test_relu_does_not_erase_gradient_from_other_paths():
    """`=` instead of `+=` makes the dead branch zero the whole node."""
    x = Value(-2.0)
    y = x.relu() + x

    y.backward()

    assert x.grad == pytest.approx(1.0)


def test_relu_accumulates_on_the_live_branch():
    a = Value(3.0)
    b = a.relu() + a

    b.backward()

    assert a.grad == pytest.approx(2.0)


def test_relu_accumulates_when_reached_twice():
    z = Value(2.0)
    w = z.relu() + z.relu()

    w.backward()

    assert z.grad == pytest.approx(2.0)


def test_relu_chains_through_other_operations():
    """x*x is positive, so the relu is live and the mul gradient passes."""
    m = Value(-1.5)
    n = (m * m).relu()

    n.backward()

    assert m.grad == pytest.approx(-3.0)

