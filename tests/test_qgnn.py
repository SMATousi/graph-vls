import numpy as np
import torch
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient

from gvls.models.qgnn import (
    QGNNClassifier,
    build_qgnn_circuit,
    per_qubit_z_observables,
    sum_z_observable,
)

M = 4
D = 8


# ── Circuit shape ────────────────────────────────────────────────────────────

def test_circuit_has_exactly_m_qubits() -> None:
    for m in (2, 4, 6, 8):
        qc, _params = build_qgnn_circuit(m, num_layers=1)
        assert qc.num_qubits == m


def test_weight_and_input_param_counts() -> None:
    m, num_layers = 4, 2
    _qc, params = build_qgnn_circuit(m, num_layers)
    # weight = num_layers * (theta + m biases) + m readout rotations
    assert len(params.weight_params) == num_layers * (1 + m) + m
    # input = m*(m-1)/2 edges + num_layers * m re-uploaded features
    assert len(params.input_params) == m * (m - 1) // 2 + num_layers * m


def test_edge_pairs_canonical_order() -> None:
    _qc, params = build_qgnn_circuit(4, num_layers=1)
    assert params.edge_pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# ── Topology equivariance (core claim of Design Decision 2) ─────────────────

def _bound_rzz_angles(m: int, num_layers: int, a_z: np.ndarray) -> dict[tuple[int, int], complex]:
    """Bind a toy A_z (thetas/biases/features/readout fixed) and return each
    edge pair's effective bound RZZ angle, keyed by qubit pair."""
    qc, params = build_qgnn_circuit(m, num_layers)
    rng = np.random.default_rng(0)
    bindings = {}
    for layer in range(num_layers):
        bindings[params.theta_params[layer]] = float(rng.uniform(0.5, 1.5))
        for i in range(m):
            bindings[params.bias_params[layer][i]] = float(rng.uniform(-1, 1))
            bindings[params.feature_params[layer][i]] = float(rng.uniform(-1, 1))
    for i in range(m):
        bindings[params.readout_params[i]] = float(rng.uniform(-1, 1))
    for idx, (i, j) in enumerate(params.edge_pairs):
        bindings[params.edge_params[idx]] = float(a_z[i, j])

    bound = qc.assign_parameters(bindings)
    angles: dict[tuple[int, int], complex] = {}
    for instruction in bound.data:
        if instruction.operation.name == "rzz":
            qubit_indices = tuple(sorted(bound.find_bit(q).index for q in instruction.qubits))
            angles[qubit_indices] = instruction.operation.params[0]
    return angles


def test_rzz_angle_nonzero_exactly_on_real_edges() -> None:
    m = 4
    a_z = np.zeros((m, m))
    a_z[0, 1] = a_z[1, 0] = 0.8
    a_z[2, 3] = a_z[3, 2] = 1.0
    # (0,2),(0,3),(1,2),(1,3) are non-edges

    angles = _bound_rzz_angles(m, num_layers=1, a_z=a_z)
    assert angles[(0, 1)] != 0
    assert angles[(2, 3)] != 0
    assert angles[(0, 2)] == 0
    assert angles[(0, 3)] == 0
    assert angles[(1, 2)] == 0
    assert angles[(1, 3)] == 0


def test_zero_a_z_all_rzz_angles_are_zero() -> None:
    m = 4
    angles = _bound_rzz_angles(m, num_layers=1, a_z=np.zeros((m, m)))
    assert all(angle == 0 for angle in angles.values())


def test_zero_a_z_reduces_to_no_entangling_reference_circuit() -> None:
    """Functional check: with A_z all-zero, the ansatz's output must match a
    hand-built reference circuit that has no RZZ gates at all (RZZ(0) is
    exactly the identity, so this is a real equivalence, not just a
    structural coincidence)."""
    m, num_layers = 3, 1
    qc, params = build_qgnn_circuit(m, num_layers)
    rng = np.random.default_rng(1)
    bindings = {}
    for layer in range(num_layers):
        bindings[params.theta_params[layer]] = float(rng.uniform(0.5, 1.5))
        for i in range(m):
            bindings[params.bias_params[layer][i]] = float(rng.uniform(-1, 1))
            bindings[params.feature_params[layer][i]] = float(rng.uniform(-1, 1))
    for i in range(m):
        bindings[params.readout_params[i]] = float(rng.uniform(-1, 1))
    for idx in range(len(params.edge_params)):
        bindings[params.edge_params[idx]] = 0.0

    bound_ansatz = qc.assign_parameters(bindings)

    from qiskit.circuit import QuantumCircuit

    reference = QuantumCircuit(m)
    for i in range(m):
        reference.ry(bindings[params.feature_params[0][i]], i)
    for i in range(m):
        reference.rz(bindings[params.bias_params[0][i]], i)
    for i in range(m):
        reference.ry(bindings[params.readout_params[i]], i)

    sv_ansatz = Statevector(bound_ansatz)
    sv_reference = Statevector(reference)
    assert sv_ansatz.equiv(sv_reference)
    np.testing.assert_allclose(sv_ansatz.data, sv_reference.data, atol=1e-10)


# ── sum_z_observable / per_qubit_z_observables ───────────────────────────────

def test_sum_z_observable_qubit_count() -> None:
    obs = sum_z_observable(5)
    assert obs.num_qubits == 5


def test_per_qubit_z_observables_returns_one_per_qubit() -> None:
    obs = per_qubit_z_observables(5)
    assert len(obs) == 5
    assert all(o.num_qubits == 5 for o in obs)


# ── QGNNClassifier: gradient flow through TorchConnector ────────────────────
# Pinned to gradient_method="param_shift" (not the SPSA default) deliberately:
# these test that each INDIVIDUAL parameter has a nonzero analytic gradient
# (the T4.4 diagonal-ansatz degeneracy bug). SPSA perturbs every differentiated
# parameter jointly along one random direction, so it cannot isolate a single
# parameter's own gradient -- under SPSA these would "pass" even if the
# ansatz were still degenerate, silently defeating the point of the test. See
# qgnn.py's module docstring, "Tradeoff, stated plainly" section.

def test_gradients_flow_to_weight_params() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="param_shift")
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0
    a_z[2, 3] = a_z[3, 2] = 1.0

    logit = model(z_tilde, a_z)
    logit.backward()

    grad = model.connector.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


def test_gradients_flow_with_multiple_layers() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=2, seed=0, gradient_method="param_shift")
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    logit = model(z_tilde, a_z)
    logit.backward()

    grad = model.connector.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0
    # every individual weight (both layers' theta/bias, plus the readout
    # rotation) must receive a nonzero gradient -- confirms the final
    # readout-rotation fix propagates gradient to every layer, not just the
    # last one (see module docstring: without it, ALL of these are zero).
    assert (grad.abs() > 1e-8).all()


def test_forward_output_is_scalar_logit() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    logit = model(z_tilde, a_z)
    assert logit.numel() == 1


def test_forward_is_deterministic_given_fixed_weights() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0
    out1 = model(z_tilde, a_z).item()
    out2 = model(z_tilde, a_z).item()
    assert out1 == out2  # exact (noiseless) simulation, no shot-noise variance


def test_different_a_z_gives_different_output() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
    torch.manual_seed(0)
    model.connector.weight.data = torch.rand(len(model.circuit_params.weight_params))
    z_tilde = torch.randn(M, D)

    a_z_empty = torch.zeros(M, M)
    a_z_edge = torch.zeros(M, M)
    a_z_edge[0, 1] = a_z_edge[1, 0] = 1.0

    out_empty = model(z_tilde, a_z_empty).item()
    out_edge = model(z_tilde, a_z_edge).item()
    assert out_empty != out_edge


# ── Batched forward/backward (T4.8) ─────────────────────────────────────────
# These are the gradient-parity checks required before trusting the batched
# path over the known-correct per-jet loop it replaces (plan.md Design
# Decision 10, requirements.md NFR-3): this codebase has already found two
# non-obvious correctness bugs in this exact pinned qiskit/qiskit-machine-
# learning stack, so reading the library source is not sufficient on its own.

def _random_batch(batch_size: int, m: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    z_tildes = torch.randn(batch_size, m, d, generator=gen)
    a_zs = torch.zeros(batch_size, m, m)
    for b in range(batch_size):
        i, j = b % m, (b + 1) % m
        a_zs[b, i, j] = a_zs[b, j, i] = 1.0
    return z_tildes, a_zs


def test_forward_batch_matches_per_jet_loop() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=3)
    batch_size = 5
    z_tildes, a_zs = _random_batch(batch_size, M, D, seed=1)

    batched_logits = model(z_tildes, a_zs)
    per_jet_logits = torch.stack(
        [model(z_tildes[b], a_zs[b]) for b in range(batch_size)]
    ).squeeze(-1)

    assert batched_logits.shape == (batch_size,)
    assert torch.allclose(batched_logits, per_jet_logits, atol=1e-6)


def test_batched_backward_matches_accumulated_per_jet_gradients() -> None:
    # gradient_method="param_shift": this is an exact-gradient equality check,
    # deliberately not exercising the (now-default) SPSA path, whose stochastic
    # per-call perturbation makes "batched == accumulated per-jet" a much
    # fuzzier, RNG-draw-order-dependent claim rather than a clean invariant.
    import torch.nn.functional as F

    batch_size = 4
    z_tildes, a_zs = _random_batch(batch_size, M, D, seed=2)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    model_batched = QGNNClassifier(m=M, d=D, num_layers=1, seed=11, gradient_method="param_shift")
    model_perjet = QGNNClassifier(m=M, d=D, num_layers=1, seed=11, gradient_method="param_shift")
    for p_b, p_j in zip(model_batched.parameters(), model_perjet.parameters()):
        assert torch.equal(p_b, p_j)  # same seed -> identical initial weights

    batched_logits = model_batched(z_tildes, a_zs)
    loss_batched = F.binary_cross_entropy_with_logits(batched_logits, labels)
    loss_batched.backward()
    grad_batched = model_batched.connector.weight.grad.clone()

    for b in range(batch_size):
        logit = model_perjet(z_tildes[b], a_zs[b])
        loss = F.binary_cross_entropy_with_logits(logit, labels[b : b + 1])
        (loss / batch_size).backward()
    grad_perjet = model_perjet.connector.weight.grad.clone()

    assert grad_batched is not None and grad_batched.abs().sum().item() > 0
    assert torch.allclose(grad_batched, grad_perjet, atol=1e-5)


def test_batch_of_one_matches_single_jet_call() -> None:
    """The ragged-final-minibatch case: a batch of size 1 must behave exactly
    like the plain single-jet call, not hit some batch-size-1 edge case."""
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=5)
    z_tildes, a_zs = _random_batch(1, M, D, seed=4)

    batched = model(z_tildes, a_zs)
    single = model(z_tildes[0], a_zs[0])

    assert batched.shape == (1,)
    assert torch.allclose(batched, single, atol=1e-6)


# ── SPSA gradient estimator (T4.8-followup) ─────────────────────────────────

def test_spsa_is_the_default_gradient_method() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
    assert model.gradient_method == "spsa"
    assert isinstance(model.connector.neural_network.gradient, SPSAEstimatorGradient)


def test_param_shift_still_selectable() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="param_shift")
    assert model.gradient_method == "param_shift"
    assert isinstance(model.connector.neural_network.gradient, ParamShiftEstimatorGradient)


def test_invalid_gradient_method_raises() -> None:
    try:
        QGNNClassifier(m=M, d=D, num_layers=1, gradient_method="finite_difference")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _backward_pub_count(model: QGNNClassifier, z_tilde: torch.Tensor, a_z: torch.Tensor) -> int:
    """Count how many (circuit, observable, params) pubs the underlying
    AerEstimatorV2 actually evaluates during one .backward() call -- a direct,
    concrete check of the evaluation-count claim in qgnn.py's module
    docstring, not just trust in the qiskit-machine-learning source."""
    logit = model(z_tilde, a_z)  # forward pass (not counted) primes the graph
    estimator = model.connector.neural_network.estimator
    original_run = estimator.run
    count = {"n": 0}

    def wrapped(pubs, **kwargs):
        pubs = list(pubs)
        count["n"] += len(pubs)
        return original_run(pubs, **kwargs)

    estimator.run = wrapped
    try:
        logit.backward()
    finally:
        estimator.run = original_run
    return count["n"]


def test_spsa_evaluation_count_is_constant_independent_of_weight_count() -> None:
    """Empirically measured, not hand-derived: param-shift's cost is NOT simply
    2*len(weight_params) here, because `theta` (one shared scalar per layer,
    QGNNCircuitParams docstring) appears in every one of the layer's
    m*(m-1)/2 RZZ gates -- the shift rule needs its own shifted pair per
    occurrence, not per parameter, so `theta` alone costs 2*num_edges
    evaluations (12 of the 28 total below, for m=4). SPSA is unaffected by
    this since it perturbs every weight jointly in one shot regardless of how
    many gates each one touches."""
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    model_ps = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="param_shift")
    model_spsa = QGNNClassifier(
        m=M, d=D, num_layers=1, seed=0, gradient_method="spsa", spsa_batch_size=1
    )

    n_ps = _backward_pub_count(model_ps, z_tilde, a_z)
    n_spsa = _backward_pub_count(model_spsa, z_tilde, a_z)

    assert n_ps == 28  # measured; see docstring for why this isn't 2*9
    assert n_spsa == 2  # spsa (batch_size=1): constant, regardless of weight/edge count
    assert n_spsa < n_ps


def test_spsa_evaluation_count_scales_with_num_layers_for_param_shift_only() -> None:
    """The gap between the two methods widens as the ansatz grows -- SPSA's
    cost is flat, param-shift's is linear in the number of weight params."""
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    model_ps_1 = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="param_shift")
    model_ps_2 = QGNNClassifier(m=M, d=D, num_layers=2, seed=0, gradient_method="param_shift")
    model_spsa_1 = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="spsa")
    model_spsa_2 = QGNNClassifier(m=M, d=D, num_layers=2, seed=0, gradient_method="spsa")

    n_ps_1 = _backward_pub_count(model_ps_1, z_tilde, a_z)
    n_ps_2 = _backward_pub_count(model_ps_2, z_tilde, a_z)
    n_spsa_1 = _backward_pub_count(model_spsa_1, z_tilde, a_z)
    n_spsa_2 = _backward_pub_count(model_spsa_2, z_tilde, a_z)

    assert n_ps_2 > n_ps_1  # more weight params at num_layers=2 -> more evaluations
    assert n_spsa_2 == n_spsa_1  # SPSA's cost is unaffected by weight-param count


def test_spsa_gradient_deterministic_given_fixed_seed() -> None:
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    model_a = QGNNClassifier(m=M, d=D, num_layers=1, seed=7, gradient_method="spsa")
    model_b = QGNNClassifier(m=M, d=D, num_layers=1, seed=7, gradient_method="spsa")

    model_a(z_tilde, a_z).backward()
    model_b(z_tilde, a_z).backward()

    assert torch.allclose(model_a.connector.weight.grad, model_b.connector.weight.grad)


def test_spsa_gradients_flow_and_are_nonzero() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="spsa")
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    model(z_tilde, a_z).backward()

    grad = model.connector.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


# ── readout_mode="learned" (T4.10 followup, validation.md V-11) ─────────────

def test_sum_mode_is_default_and_has_no_readout_head() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
    assert model.readout_mode == "sum"
    assert model.readout_head is None


def test_learned_mode_has_readout_head_with_expected_shape() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, readout_mode="learned")
    assert model.readout_head is not None
    assert model.readout_head.weight.shape == (1, M)
    assert model.readout_head.bias.shape == (1,)


def test_invalid_readout_mode_raises() -> None:
    try:
        QGNNClassifier(m=M, d=D, num_layers=1, readout_mode="weighted_sum")
        raise AssertionError("expected ValueError for an unknown readout_mode")
    except ValueError as exc:
        assert "weighted_sum" in str(exc)


def test_learned_mode_forward_shapes_single_and_batched() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, readout_mode="learned")
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0
    single = model(z_tilde, a_z)
    assert single.shape == (1,)

    z_tildes, a_zs = _random_batch(3, M, D, seed=1)
    batched = model(z_tildes, a_zs)
    assert batched.shape == (3,)


def test_learned_mode_gradients_flow_to_both_quantum_and_classical_weights() -> None:
    model = QGNNClassifier(
        m=M, d=D, num_layers=1, seed=0, gradient_method="param_shift", readout_mode="learned"
    )
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    logit = model(z_tilde, a_z)
    logit.backward()

    quantum_grad = model.connector.weight.grad
    assert quantum_grad is not None
    assert quantum_grad.abs().sum().item() > 0

    assert model.readout_head.weight.grad is not None
    assert model.readout_head.weight.grad.abs().sum().item() > 0
    assert model.readout_head.bias.grad is not None
    assert model.readout_head.bias.grad.abs().sum().item() > 0


def test_learned_mode_deterministic_given_fixed_seed() -> None:
    model_a = QGNNClassifier(m=M, d=D, num_layers=1, seed=3, readout_mode="learned")
    model_b = QGNNClassifier(m=M, d=D, num_layers=1, seed=3, readout_mode="learned")
    assert torch.allclose(model_a.connector.weight, model_b.connector.weight)
    assert torch.allclose(model_a.readout_head.weight, model_b.readout_head.weight)
    assert torch.allclose(model_a.readout_head.bias, model_b.readout_head.bias)

    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0
    assert torch.allclose(model_a(z_tilde, a_z), model_b(z_tilde, a_z))


def test_learned_mode_multiplies_circuit_evaluations_by_observable_count() -> None:
    """Empirically measured, not assumed: EstimatorQNN's gradient estimators
    submit one pub per observable rather than batching a multi-observable
    readout into a single pub, so "learned" mode's per-step circuit-
    evaluation cost is `sum`'s cost times m (num observables) -- for both
    SPSA and param-shift. Cheap in absolute terms at m=4 (2->8), but a real,
    measured cost, not the "free" readout an exact-statevector single state
    preparation might suggest."""
    z_tilde = torch.randn(M, D)
    a_z = torch.zeros(M, M)
    a_z[0, 1] = a_z[1, 0] = 1.0

    model_sum_spsa = QGNNClassifier(m=M, d=D, num_layers=1, seed=0, gradient_method="spsa")
    model_learned_spsa = QGNNClassifier(
        m=M, d=D, num_layers=1, seed=0, gradient_method="spsa", readout_mode="learned"
    )
    n_sum = _backward_pub_count(model_sum_spsa, z_tilde, a_z)
    n_learned = _backward_pub_count(model_learned_spsa, z_tilde, a_z)

    assert n_sum == 2
    assert n_learned == n_sum * M
