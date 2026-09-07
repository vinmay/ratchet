"""Tests for the network layers built on top of the engine."""

import math

import pytest

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Neuron, mse_loss


def test_neuron_has_one_weight_per_input_plus_a_bias():
    assert len(Neuron(1).parameters()) == 2
    assert len(Neuron(2).parameters()) == 3
    assert len(Neuron(5).parameters()) == 6


def test_neuron_parameters_are_values():
    """Floats cannot carry gradients, so parameters must be nodes."""
    n = Neuron(3)

    assert all(isinstance(p, Value) for p in n.parameters())


def test_neuron_parameters_are_the_same_objects_the_graph_uses():
    """The optimiser mutates these in place; copies would break training."""
    n = Neuron(2)
    params = n.parameters()

    assert params[0] is n.weights[0]
    assert any(p is n.bias for p in params)


def test_neuron_parameters_does_not_mutate_the_neuron():
    """A query must not change state. `weights.append(bias)` fails this."""
    n = Neuron(2)

    n.parameters()
    n.parameters()
    n.parameters()

    assert len(n.weights) == 2
    assert len(n.parameters()) == 3


def test_neuron_weights_are_not_all_identical():
    """Zero or constant init is a Unit 4 failure, not a starting point."""
    n = Neuron(8)
    values = [w.data for w in n.weights]

    assert len(set(values)) > 1
    assert all(-1.0 <= v <= 1.0 for v in values)


def test_neuron_forward_returns_a_value():
    n = Neuron(2)
    out = n([2.0, 3.0])

    assert isinstance(out, Value)


def test_neuron_forward_matches_the_arithmetic_by_hand():
    n = Neuron(2)
    x = [2.0, 3.0]

    out = n(x)

    w0, w1, b = n.weights[0].data, n.weights[1].data, n.bias.data
    assert out.data == pytest.approx(math.tanh(w0 * x[0] + w1 * x[1] + b))


def test_neuron_output_is_squashed_by_the_activation():
    n = Neuron(2)
    out = n([100.0, 100.0])

    assert -1.0 <= out.data <= 1.0


def test_neuron_backward_reaches_every_parameter():
    """A break in the graph shows up here as a zero gradient, not an error."""
    n = Neuron(2)
    out = n([2.0, 3.0])

    out.backward()

    assert all(p.grad != 0.0 for p in n.parameters())


def test_neuron_weight_gradient_is_the_input_times_the_activation_slope():
    n = Neuron(2)
    x = [2.0, 3.0]
    out = n(x)

    out.backward()

    slope = 1 - out.data ** 2
    assert n.weights[0].grad == pytest.approx(x[0] * slope)
    assert n.weights[1].grad == pytest.approx(x[1] * slope)
    assert n.bias.grad == pytest.approx(slope)


def test_neuron_gradients_accumulate_across_two_forward_passes():
    """backward() does not zero. The training loop must."""
    n = Neuron(2)

    n([1.0, 1.0]).backward()
    first = n.bias.grad
    n([1.0, 1.0]).backward()

    assert n.bias.grad == pytest.approx(2 * first)


# ---------------------------------------------------------------------------
# Layer: nout neurons, each taking nin inputs. Same input to all of them.
# ---------------------------------------------------------------------------


def test_layer_returns_one_output_per_neuron():
    outs = Layer(2, 3)([2.0, 3.0])

    assert isinstance(outs, list)
    assert len(outs) == 3
    assert all(isinstance(o, Value) for o in outs)


def test_layer_output_is_flat_not_nested():
    """`append([n(x)])` gives a list of one-element lists. The next layer breaks."""
    outs = Layer(2, 3)([2.0, 3.0])

    assert not any(isinstance(o, list) for o in outs)


def test_layer_parameter_count():
    assert len(Layer(2, 3).parameters()) == 9      # 3 x (2 weights + 1 bias)
    assert len(Layer(4, 1).parameters()) == 5
    assert len(Layer(3, 5).parameters()) == 20


def test_layer_parameters_are_flat_values():
    params = Layer(2, 3).parameters()

    assert all(isinstance(p, Value) for p in params)


def test_layer_parameters_does_not_mutate():
    layer = Layer(2, 3)

    layer.parameters()
    layer.parameters()

    assert len(layer.parameters()) == 9
    assert len(layer.neurons) == 3


def test_layer_parameters_excludes_activations():
    """Outputs are rebuilt every call; only weights and biases are trained."""
    layer = Layer(2, 3)

    before = len(layer.parameters())
    layer([2.0, 3.0])

    assert len(layer.parameters()) == before


def test_layer_neurons_are_distinct_objects():
    """Reusing one neuron object would train every slot identically."""
    layer = Layer(2, 3)

    assert layer.neurons[0] is not layer.neurons[1]
    assert layer.neurons[0].weights[0] is not layer.neurons[1].weights[0]


def test_layer_neurons_persist_across_calls():
    """Building neurons in __call__ would discard everything the optimiser learned."""
    layer = Layer(2, 3)
    first = layer.neurons[0].weights[0]

    layer([2.0, 3.0])
    layer([1.0, 1.0])

    assert layer.neurons[0].weights[0] is first


def test_layer_backward_reaches_every_parameter():
    layer = Layer(2, 3)
    outs = layer([2.0, 3.0])

    (outs[0] + outs[1] + outs[2]).backward()

    assert all(p.grad != 0.0 for p in layer.parameters())


def test_layer_outputs_feed_the_next_layer():
    """Layer 2's nin is layer 1's nout. This is what MLP chains."""
    first = Layer(2, 3)
    second = Layer(3, 2)

    outs = second(first([2.0, 3.0]))

    assert len(outs) == 2
    assert all(isinstance(o, Value) for o in outs)


# ---------------------------------------------------------------------------
# MLP: layers chained, each one's outputs feeding the next.
# ---------------------------------------------------------------------------


def test_mlp_layer_shapes_chain():
    """Layer i's nin is layer i-1's nout. [nin] + nouts, read in pairs."""
    m = MLP(2, [4, 4, 1])

    shapes = [(len(l.neurons[0].weights), len(l.neurons)) for l in m.layers]
    assert shapes == [(2, 4), (4, 4), (4, 1)]


def test_mlp_handles_a_single_layer():
    m = MLP(2, [3])

    assert len(m.layers) == 1
    assert len(m.parameters()) == 9


def test_mlp_unwraps_a_single_output_to_a_bare_value():
    """__call__ returns out[0], so the caller does not index."""
    out = MLP(2, [4, 4, 1])([2.0, 3.0])

    assert isinstance(out, Value)
    assert not isinstance(out, list)


def test_mlp_with_several_outputs_returns_the_whole_list():
    """Unwrapping is conditional, so nothing is discarded when nout > 1."""
    out = MLP(2, [4, 3])([2.0, 3.0])

    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(o, Value) for o in out)


def test_mlp_unwrap_depends_only_on_the_last_layer():
    """A hidden layer of width 1 must not change the output shape."""
    assert isinstance(MLP(2, [1, 3])([2.0, 3.0]), list)
    assert isinstance(MLP(2, [3, 1])([2.0, 3.0]), Value)


def test_mlp_runs_layers_in_sequence_not_in_parallel():
    """Collecting per-layer outputs would return one entry per layer."""
    m = MLP(2, [4, 4, 1])

    out = m([2.0, 3.0])

    assert isinstance(out, Value)


def test_mlp_parameter_count():
    assert len(MLP(2, [4, 4, 1]).parameters()) == 37    # 12 + 20 + 5
    assert len(MLP(3, [2, 1]).parameters()) == 11       # 8 + 3


def test_mlp_parameters_are_flat_values():
    params = MLP(2, [4, 4, 1]).parameters()

    assert all(isinstance(p, Value) for p in params)
    assert not any(isinstance(p, list) for p in params)


def test_mlp_parameters_does_not_mutate():
    m = MLP(2, [4, 4, 1])

    m.parameters()
    m.parameters()

    assert len(m.parameters()) == 37


def test_mlp_parameters_are_the_nodes_the_graph_uses():
    m = MLP(2, [4, 4, 1])

    assert m.parameters()[0] is m.layers[0].neurons[0].weights[0]


def test_mlp_backward_reaches_every_parameter():
    """One break anywhere in the pipeline leaves parameters at zero."""
    m = MLP(2, [4, 4, 1])

    m([2.0, 3.0]).backward()

    assert all(p.grad != 0.0 for p in m.parameters())


def test_mlp_layers_persist_across_calls():
    m = MLP(2, [4, 4, 1])
    first = m.parameters()[0]

    m([2.0, 3.0])
    m([1.0, 1.0])

    assert m.parameters()[0] is first


# ---------------------------------------------------------------------------
# mse_loss. Must return a Value still attached to the graph: reading .data
# anywhere inside severs it, and training then silently does nothing.
# ---------------------------------------------------------------------------


def test_mse_is_zero_for_a_perfect_prediction():
    assert mse_loss([Value(1.0)], [1.0]).data == pytest.approx(0.0)


def test_mse_is_the_squared_error():
    assert mse_loss([Value(0.0)], [1.0]).data == pytest.approx(1.0)
    assert mse_loss([Value(-1.0)], [1.0]).data == pytest.approx(4.0)
    assert mse_loss([Value(0.5)], [-1.0]).data == pytest.approx(2.25)


def test_mse_averages_over_the_batch():
    """Summing without dividing makes the loss scale with dataset size."""
    assert mse_loss([Value(0.0), Value(1.0)], [1.0, 1.0]).data == pytest.approx(0.5)
    assert mse_loss([Value(0.0)] * 4, [1.0] * 4).data == pytest.approx(1.0)


def test_mse_is_symmetric_in_the_sign_of_the_error():
    over = mse_loss([Value(1.5)], [1.0]).data
    under = mse_loss([Value(0.5)], [1.0]).data

    assert over == pytest.approx(under)


def test_mse_returns_a_value_not_a_float():
    loss = mse_loss([Value(0.0)], [1.0])

    assert isinstance(loss, Value)


def test_mse_backward_reaches_every_model_parameter():
    """0/37 here means the loss rebuilt a fresh node and dropped the graph."""
    m = MLP(2, [4, 4, 1])
    X = [[2.0, 3.0], [-1.0, 0.5], [0.0, 0.0]]
    y = [1.0, -1.0, 1.0]

    preds = [m(x) for x in X]
    mse_loss(preds, y).backward()

    assert all(p.grad != 0.0 for p in m.parameters())


def test_mse_gradient_points_downhill():
    """One SGD step on a fixed input must reduce the loss."""
    m = MLP(2, [4, 4, 1])
    x, target = [2.0, 3.0], 1.0

    before = mse_loss([m(x)], [target])
    before.backward()
    for p in m.parameters():
        p.data -= 0.01 * p.grad

    after = mse_loss([m(x)], [target])

    assert after.data < before.data


# ---------------------------------------------------------------------------
# zero_grad. backward() accumulates by design, so resetting is the caller's job.
# ---------------------------------------------------------------------------


def test_zero_grad_clears_every_parameter():
    m = MLP(2, [4, 4, 1])
    m([2.0, 3.0]).backward()

    m.zero_grad()

    assert all(p.grad == 0.0 for p in m.parameters())


def test_zero_grad_leaves_the_weights_alone():
    """It resets gradients, not parameters."""
    m = MLP(2, [4, 4, 1])
    before = [p.data for p in m.parameters()]

    m.zero_grad()

    assert [p.data for p in m.parameters()] == before


def test_repeated_backward_gives_the_same_gradient_when_zeroed():
    """Without this, the effective learning rate grows every epoch."""
    m = MLP(2, [4, 4, 1])
    x, target = [2.0, 3.0], 1.0
    p = m.parameters()[0]

    grads = []
    for _ in range(3):
        m.zero_grad()
        mse_loss([m(x)], [target]).backward()
        grads.append(p.grad)

    assert grads[1] == pytest.approx(grads[0])
    assert grads[2] == pytest.approx(grads[0])


def test_without_zero_grad_gradients_pile_up():
    """Documents why the loop needs it. If this fails, backward() grew a reset."""
    m = MLP(2, [4, 4, 1])
    x, target = [2.0, 3.0], 1.0
    p = m.parameters()[0]

    mse_loss([m(x)], [target]).backward()
    first = p.grad
    mse_loss([m(x)], [target]).backward()

    assert p.grad == pytest.approx(2 * first)


def test_zero_grad_is_safe_before_any_backward():
    m = MLP(2, [4, 4, 1])

    m.zero_grad()

    assert all(p.grad == 0.0 for p in m.parameters())

