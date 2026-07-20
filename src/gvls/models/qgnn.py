"""Verdon-style Quantum Graph Neural Network ansatz (T4.4).

One qubit per pooled latent node (`M` qubits); entangling `RZZ` gates encode
the learned latent graph `A_z`, `RZ` gates give each qubit a trainable bias,
and `RY` gates re-upload `z_tilde`'s per-node features across layers (Verdon,
Broughton, McClean et al., "Quantum Graph Neural Networks," arXiv:1909.12264;
`specs/phase4/plan.md` Design Decision 2).

**Fixed maximal-topology circuit, not a per-jet rebuild.** `plan.md` describes
`build_qgnn_circuit` taking `(M, d, num_layers)`; `A_z`'s edges could instead
have been used *structurally* (only emitting an `RZZ` instruction for real
edges), which would require constructing a brand-new circuit -- and therefore
a brand-new `EstimatorQNN`/`TorchConnector` -- for every jet, since jets have
different latent-graph topologies. That conflicts with how `TorchConnector`
actually manages trainable weights: it owns its weight tensor as its own
`nn.Parameter`, created fresh (via `torch.tensor(initial_weights)`, which
does not preserve an autograd link) every time a `TorchConnector` is
instantiated. Rebuilding it per jet would mean `theta`/`b_i`'s gradients
never reach one persistent, Adam-optimized parameter without manually
relaying gradients between successive fresh copies -- effectively
reimplementing part of what `TorchConnector` already does, contradicting
`plan.md`'s explicit "no custom backward pass needs to be written" intent.

Instead, the circuit spans *every* possible qubit pair once, with each pair's
`RZZ` angle equal to `theta_l * A_z[i,j]` -- `A_z[i,j]` bound as a per-call
*input* parameter (0 for a non-edge). Since `RZZ(0)` is exactly the identity
gate, this is functionally identical to omitting the gate for that pair; the
circuit's entangling behavior is still a direct, literal function of `A_z`
(Design Decision 2's actual requirement), just realized via zero-coefficient
binding instead of gate-object omission. This lets one `EstimatorQNN`/
`TorchConnector` pair be built once per `(M, d, num_layers)` and reused for
every jet -- exactly how `TorchConnector` is meant to be used -- with
`theta`/`b_i`/the readout rotation (see below) as its persistent,
Adam-trainable weights.

**Correctness fix found during implementation: a final non-diagonal readout
rotation is required, not optional.** `RZZ` and `RZ` are both diagonal gates
in the computational basis; a `Z`-basis measurement commutes exactly with any
diagonal unitary applied beforehand (`U^dagger Z U = Z` whenever `U` is
diagonal). A circuit built exactly as `plan.md`/FR-4 literally describe it --
`RY` data encoding, then only `RZZ`+`RZ` before measuring `sum(Z_i)` -- is
therefore provably *degenerate*: `theta` and `b_i`'s gradients are
identically zero regardless of `A_z` or the data, confirmed empirically
(`grad ~ 1e-16`, and the QNN's output was bit-identical across very different
`theta`/`b_i` values) before this fix. Appending one final trainable `RY`
rotation per qubit after all `num_layers` (a standard basis-changing readout
layer) restores a real, nonzero gradient to every layer's entangling and bias
parameters -- verified empirically (`theta`'s gradient went from ~1e-16 to
~0.12 on a toy 2-qubit circuit) before this design was adopted.

**Exact (noiseless) simulation.** `EstimatorQNN`'s own `default_precision`
(0.015625) triggers shot-based sampling even when the underlying
`qiskit_aer.primitives.EstimatorV2` is otherwise configured for exact
evaluation -- confirmed empirically: without `default_precision=0.0` passed
explicitly to `EstimatorQNN`, repeated calls with identical inputs returned
slightly different values (shot noise); with it, they were bit-identical.
FR-4 requires Aer's *noiseless statevector* simulator, so `default_precision=0.0`
is set explicitly here rather than relying on the estimator's own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import Tensor


@dataclass
class QGNNCircuitParams:
    """Every `Parameter` in a `build_qgnn_circuit` circuit, grouped by role."""

    edge_pairs: list[tuple[int, int]]      # canonical (i < j) qubit-pair order
    edge_params: list[Parameter]           # A_z[i,j] per pair, in edge_pairs order
    feature_params: list[list[Parameter]]  # [layer][qubit]: z_tilde re-uploading
    theta_params: list[Parameter]          # [layer]: one shared scalar per layer
    bias_params: list[list[Parameter]]     # [layer][qubit]
    readout_params: list[Parameter]        # [qubit]: final non-diagonal rotation

    @property
    def weight_params(self) -> list[Parameter]:
        """Trainable weights, in the order EstimatorQNN's weight_params expects."""
        params: list[Parameter] = []
        for layer in range(len(self.theta_params)):
            params.append(self.theta_params[layer])
            params.extend(self.bias_params[layer])
        params.extend(self.readout_params)
        return params

    @property
    def input_params(self) -> list[Parameter]:
        """Runtime inputs (A_z, then z_tilde), in the order forward() must bind them."""
        params: list[Parameter] = list(self.edge_params)
        for layer in self.feature_params:
            params.extend(layer)
        return params


def sum_z_observable(m: int) -> SparsePauliOp:
    """Sum of single-qubit Z observables across all m qubits.

    Chosen over a single designated readout qubit (FR-4's other option) so
    every pooled latent node contributes to the classification signal, not
    just one arbitrarily chosen qubit -- consistent with this project's
    general stance that all M pooled nodes matter, not one distinguished
    node. Not empirically compared against the single-readout-qubit
    alternative (that comparison is only decidable once T4.5 trains a real
    classifier); documented here as the chosen default per FR-4's explicit
    "pick whichever ... record the choice."
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return SparsePauliOp.from_sparse_list([("Z", [i], 1.0) for i in range(m)], num_qubits=m)


def build_qgnn_circuit(m: int, num_layers: int = 1) -> tuple[QuantumCircuit, QGNNCircuitParams]:
    """Verdon-style QGNN ansatz circuit (T4.4, FR-4).

    Per layer: an RY data-encoding rotation per qubit (re-uploading -- see
    `QGNNClassifier.encode_input` for which z_tilde dimension each layer
    uses), an RZZ(theta_layer * A_z[i,j]) entangling gate for every possible
    qubit pair, and an RZ(b_i) bias rotation per qubit. After all layers, one
    final RY(gamma_i) readout rotation per qubit (see module docstring: this
    final non-diagonal layer is a correctness requirement, not decoration).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    edge_pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    edge_params = [Parameter(f"a_{i}_{j}") for i, j in edge_pairs]
    feature_params = [
        [Parameter(f"x_{layer}_{i}") for i in range(m)] for layer in range(num_layers)
    ]
    theta_params = [Parameter(f"theta_{layer}") for layer in range(num_layers)]
    bias_params = [[Parameter(f"b_{layer}_{i}") for i in range(m)] for layer in range(num_layers)]
    readout_params = [Parameter(f"g_{i}") for i in range(m)]

    qc = QuantumCircuit(m)
    for layer in range(num_layers):
        for i in range(m):
            qc.ry(feature_params[layer][i], i)
        for (i, j), a_ij in zip(edge_pairs, edge_params):
            qc.rzz(theta_params[layer] * a_ij, i, j)
        for i in range(m):
            qc.rz(bias_params[layer][i], i)
    for i in range(m):
        qc.ry(readout_params[i], i)

    params = QGNNCircuitParams(
        edge_pairs=edge_pairs,
        edge_params=edge_params,
        feature_params=feature_params,
        theta_params=theta_params,
        bias_params=bias_params,
        readout_params=readout_params,
    )
    return qc, params


class QGNNClassifier(nn.Module):
    """Verdon-style QGNN readout on a pooled GVLS latent graph (T4.4).

    `forward(z_tilde, a_z) -> Tensor`: a single logit (raw expectation value
    of `sum_z_observable`, fed to a BCE-with-logits loss in T4.5 -- consistent
    with the rest of this codebase's convention of working with logits, not
    probabilities, everywhere). Built once per `(m, d, num_layers)` and
    reused for every jet (see module docstring for why); only `theta`, `b_i`,
    and the final readout rotation are trainable -- `z_tilde` and `A_z` are
    runtime inputs, never weights.
    """

    def __init__(self, m: int, d: int, num_layers: int = 1, seed: int | None = None) -> None:
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.m = m
        self.d = d
        self.num_layers = num_layers

        circuit, self.circuit_params = build_qgnn_circuit(m, num_layers)
        observable = sum_z_observable(m)
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=self.circuit_params.input_params,
            weight_params=self.circuit_params.weight_params,
            input_gradients=True,
            estimator=AerEstimatorV2(),
            default_precision=0.0,  # exact statevector expectation, no shot noise
        )

        initial_weights = None
        if seed is not None:
            rng = np.random.default_rng(seed)
            initial_weights = rng.uniform(-0.1, 0.1, size=len(self.circuit_params.weight_params))
        self.connector = TorchConnector(qnn, initial_weights=initial_weights)

    def encode_input(self, z_tilde: Tensor, a_z: Tensor) -> Tensor:
        """Flatten (z_tilde, a_z) into the circuit's input-parameter order:
        edge values (canonical i<j order) first, then re-uploaded features
        (layer-major: layer 0's m qubits, then layer 1's, ...), matching
        `QGNNCircuitParams.input_params`'s order exactly.
        """
        values: list[Tensor] = [a_z[i, j] for i, j in self.circuit_params.edge_pairs]
        for layer in range(self.num_layers):
            dim = layer % self.d
            values.extend(z_tilde[i, dim] for i in range(self.m))
        return torch.stack(values).float()

    def forward(self, z_tilde: Tensor, a_z: Tensor) -> Tensor:
        return self.connector(self.encode_input(z_tilde, a_z))
