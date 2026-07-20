import numpy as np
import torch
from qiskit.quantum_info import Statevector

from gvls.models.qgnn import QGNNClassifier, build_qgnn_circuit, sum_z_observable

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


# ── sum_z_observable ─────────────────────────────────────────────────────────

def test_sum_z_observable_qubit_count() -> None:
    obs = sum_z_observable(5)
    assert obs.num_qubits == 5


# ── QGNNClassifier: gradient flow through TorchConnector ────────────────────

def test_gradients_flow_to_weight_params() -> None:
    model = QGNNClassifier(m=M, d=D, num_layers=1, seed=0)
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
    model = QGNNClassifier(m=M, d=D, num_layers=2, seed=0)
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
