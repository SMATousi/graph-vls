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

**Batched training (T4.8, 2026-07-27).** `forward` accepts either a single
jet (`z_tilde`: `(M,d)`, `a_z`: `(M,M)`) or a minibatch (`(B,M,d)`, `(B,M,M)`),
dispatching on `z_tilde.dim()`. Batching routes through one `TorchConnector`
call for the whole minibatch instead of one call per jet -- verified directly
against the installed `qiskit-machine-learning` source (`connectors/
torch_connector.py`, `neural_networks/estimator_qnn.py`) that a 2D
`(B, num_inputs)` input with a single shared weight vector is natively
supported and submitted as one Estimator job regardless of `B`
(`specs/phase4/plan.md` Design Decision 10). This does not reduce the number
of parameter-shift circuit evaluations (`O(B x num_params)` either way); it
removes `B`-fold per-jet job-dispatch overhead. `input_gradients` is now
`False`: `z_tilde`/`a_z` are frozen extraction outputs that never need
`d(loss)/d(input)` (T4.5's two-stage design freezes GVLS before the QGNN is
ever trained), so computing them via parameter-shift was pure waste --
roughly doubling the shifted-circuit count for no benefit.

Measuring T4.8 directly (`specs/phase4/validation.md` V-8) showed batching's
own contribution was small (~1.07-1.08x): most circuit-evaluation cost is not
per-jet Python dispatch overhead, it's `ParamShiftEstimatorGradient` itself
needing shifted circuit evaluations per differentiated parameter, per sample
(`weight_params` only, since `input_gradients=False`).

**SPSA gradient estimator (T4.8-followup, 2026-07-27) -- an actual reduction
in evaluation count, not just dispatch overhead.** `qiskit_machine_learning.
gradients.SPSAEstimatorGradient` (verified by reading `spsa_estimator_
gradient.py`, not assumed) perturbs *all* differentiated parameters at once
along one random +-1 direction and takes a single two-point finite difference
-- exactly `2 * spsa_batch_size` circuit evaluations per sample, *independent
of how many parameters are being differentiated or how many gates they touch*.
Measured directly (`tests/test_qgnn.py::test_spsa_evaluation_count_is_
constant_independent_of_weight_count`, not assumed from theory): for the
current `m=4, num_layers=1` ansatz, `ParamShiftEstimatorGradient` needs `28`
circuit evaluations per sample versus SPSA's constant `2` at the default
`spsa_batch_size=1` -- a `14x` reduction (`24x` at `num_layers=2`, `48` vs.
`2` -- the ratio widens as the ansatz grows, since SPSA's cost never
changes). Note `28`, not `2*9`: `theta` (one shared scalar per layer,
`QGNNCircuitParams` docstring) appears in every one of that layer's
`m*(m-1)/2` RZZ gates, and the shift rule needs its own shifted pair per
*occurrence*, not per parameter -- `theta` alone costs `2*num_edges = 12` of
the `28`; only `bias`/`readout` (one gate each) cost the naively-expected `2`
each. `gradient_method="spsa"` is the new default (`gradient_method=
"param_shift"` remains available and is what the T4.8 gradient-parity/
degeneracy tests in `tests/test_qgnn.py` pin to explicitly -- see below for
why).

**Tradeoff, stated plainly: SPSA gradients are stochastic, not analytic.**
Each call's gradient estimate comes from one random-direction finite
difference, not an exact per-parameter derivative -- it has real variance,
and every differentiated parameter in a single call gets the *same-magnitude*
estimate (`+-diff`, differing only in the perturbation's random sign for that
parameter), since one shared scalar `diff` is divided by each parameter's own
+-1 offset. This has a specific, non-obvious consequence for testing: SPSA
cannot distinguish "this individual parameter has a genuinely zero analytic
gradient" (the exact bug T4.4 found and fixed with the readout rotation) from
"this parameter happened to matter in this call only because it was perturbed
jointly with others that do" -- a parameter-shift estimate isolates each
parameter; an SPSA estimate never does. For that reason the degeneracy-
sensitive tests (`test_gradients_flow_to_weight_params`, `test_gradients_
flow_with_multiple_layers`) and T4.8's exact gradient-parity tests
(`test_batched_backward_matches_accumulated_per_jet_gradients`) explicitly
pass `gradient_method="param_shift"` rather than relying on the (now-SPSA)
default -- they are testing a property only an exact gradient estimator can
verify. Determinism given a fixed `seed` is preserved for SPSA too (its own
internal RNG is seeded from the same `seed` `QGNNClassifier` already uses for
weight initialization), satisfying NFR-1's reproducibility requirement, but
"deterministic" here means "the same stochastic estimate every time for the
same seed and inputs," not "the true analytic gradient."
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
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import Tensor

GRADIENT_METHODS = ("spsa", "param_shift")
READOUT_MODES = ("sum", "learned")


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


def per_qubit_z_observables(m: int) -> list[SparsePauliOp]:
    """Per-qubit Z observables, one per pooled latent node (`readout_mode="learned"`).

    `sum_z_observable`'s combined operator forces every qubit's contribution
    to the final logit through a fixed, unweighted coefficient of 1.0 -- the
    trained weights can shape the state each `<Z_i>` is measured against, but
    nothing can learn "qubit 2 matters more than qubit 0" in the final
    combination. Measuring each qubit separately (`EstimatorQNN` natively
    accepts a list of observables, returning one expectation value per
    observable) lets `QGNNClassifier`'s own trainable classical
    `Linear(m, 1)` head learn that weighting instead -- the combination
    classical baselines get for free (a linear model's decision score IS a
    learned weighted combination of its inputs) but the fixed-sum readout
    structurally could not express. See `specs/phase4/validation.md` V-11.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=m) for i in range(m)]


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

    `forward(z_tilde, a_z) -> Tensor`: single-jet inputs (`(M,d)`, `(M,M)`)
    return one logit, shape `(1,)`; batched inputs (`(B,M,d)`, `(B,M,M)`)
    return one logit per jet, shape `(B,)` (T4.8), fed to a BCE-with-logits
    loss, consistent with the rest of this codebase's convention of working
    with logits, not probabilities, everywhere. Built once per
    `(m, d, num_layers)` and reused for every jet/minibatch (see module
    docstring for why); `z_tilde` and `A_z` are runtime inputs, never weights.

    `readout_mode` (T4.10 followup, validation.md V-11) selects how the `m`
    per-qubit measurements become one logit:
    - `"sum"` (default, backward compatible): `sum_z_observable`'s single
      combined operator -- every qubit's `<Z_i>` contributes with a fixed,
      unweighted coefficient of 1.0. Only `theta`, `b_i`, and the final
      readout rotation are trainable.
    - `"learned"`: `per_qubit_z_observables` measures each qubit separately
      (`EstimatorQNN` returns an `(m,)`/`(B,m)` vector instead of a scalar),
      fed through a trainable classical `nn.Linear(m, 1)` (`self.readout_head`)
      to produce the logit -- lets the model learn which pooled nodes matter
      more, the weighted-combination expressiveness a classical linear model
      has for free but the fixed-sum readout structurally lacks. This head's
      gradient is computed by ordinary PyTorch autograd (exact, not
      SPSA/parameter-shift) since it sits entirely outside `TorchConnector`,
      so it does not add to the quantum circuit's own trainable-weight count
      that SPSA's joint-perturbation gradient estimate has to spread across.
    """

    def __init__(
        self,
        m: int,
        d: int,
        num_layers: int = 1,
        seed: int | None = None,
        gradient_method: str = "spsa",
        spsa_epsilon: float = 1e-6,
        spsa_batch_size: int = 1,
        readout_mode: str = "sum",
    ) -> None:
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if gradient_method not in GRADIENT_METHODS:
            raise ValueError(
                f"gradient_method must be one of {GRADIENT_METHODS}, got {gradient_method!r}"
            )
        if readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}, got {readout_mode!r}")
        self.m = m
        self.d = d
        self.num_layers = num_layers
        self.gradient_method = gradient_method
        self.readout_mode = readout_mode

        circuit, self.circuit_params = build_qgnn_circuit(m, num_layers)
        observable = (
            per_qubit_z_observables(m) if readout_mode == "learned" else sum_z_observable(m)
        )
        estimator = AerEstimatorV2()  # default_precision=0.0 (exact) -- see class Options
        if gradient_method == "spsa":
            # 2 * spsa_batch_size circuit evaluations per sample, independent
            # of len(weight_params) -- see module docstring for why this
            # replaces param_shift as the default.
            gradient = SPSAEstimatorGradient(
                estimator=estimator, epsilon=spsa_epsilon, batch_size=spsa_batch_size, seed=seed
            )
        else:
            gradient = ParamShiftEstimatorGradient(estimator=estimator)
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=self.circuit_params.input_params,
            weight_params=self.circuit_params.weight_params,
            input_gradients=False,
            estimator=estimator,
            default_precision=0.0,  # exact statevector expectation, no shot noise
            gradient=gradient,
        )

        initial_weights = None
        rng = None
        if seed is not None:
            rng = np.random.default_rng(seed)
            initial_weights = rng.uniform(-0.1, 0.1, size=len(self.circuit_params.weight_params))
        self.connector = TorchConnector(qnn, initial_weights=initial_weights)

        self.readout_head: nn.Linear | None = None
        if readout_mode == "learned":
            self.readout_head = nn.Linear(m, 1)
            if rng is not None:
                # Continue the same deterministic numpy RNG stream used for
                # the quantum weights above (rather than relying on PyTorch's
                # own global RNG, which the caller may or may not have
                # seeded) so `seed` fully determines every trainable
                # parameter in this module, quantum and classical alike.
                head_weight = rng.uniform(-0.1, 0.1, size=(1, m))
                head_bias = rng.uniform(-0.1, 0.1, size=(1,))
                with torch.no_grad():
                    self.readout_head.weight.copy_(torch.from_numpy(head_weight).float())
                    self.readout_head.bias.copy_(torch.from_numpy(head_bias).float())

    def encode_input(self, z_tilde: Tensor, a_z: Tensor) -> Tensor:
        """Flatten one jet's (z_tilde, a_z) into the circuit's input-parameter
        order: edge values (canonical i<j order) first, then re-uploaded
        features (layer-major: layer 0's m qubits, then layer 1's, ...),
        matching `QGNNCircuitParams.input_params`'s order exactly.
        """
        values: list[Tensor] = [a_z[i, j] for i, j in self.circuit_params.edge_pairs]
        for layer in range(self.num_layers):
            dim = layer % self.d
            values.extend(z_tilde[i, dim] for i in range(self.m))
        return torch.stack(values).float()

    def encode_input_batch(self, z_tildes: Tensor, a_zs: Tensor) -> Tensor:
        """Batched counterpart of `encode_input` (T4.8): `(B,M,d)`, `(B,M,M)`
        -> `(B, num_input_params)`. Valid only because every jet is pooled to
        the same fixed `M` (Design Decision 3) -- every jet's flattened input
        has identical length, so stacking needs no padding/masking, unlike
        the classical encoder/pooling stack's per-jet-`N` constraint (Design
        Decision 7). The loop here is pure tensor indexing (no quantum
        calls), so its cost is negligible next to the batched `TorchConnector`
        call it feeds.
        """
        return torch.stack(
            [self.encode_input(z_tildes[b], a_zs[b]) for b in range(z_tildes.shape[0])]
        )

    def forward(self, z_tilde: Tensor, a_z: Tensor) -> Tensor:
        if z_tilde.dim() == 3:
            # batched: (B, num_inputs) -> TorchConnector returns (B, 1) in
            # "sum" mode or (B, m) in "learned" mode.
            raw = self.connector(self.encode_input_batch(z_tilde, a_z))
            if self.readout_head is not None:
                raw = self.readout_head(raw)
            # squeeze to (B,) to match a per-jet call's (1,)-shaped output
            # and BCE's expected label shape.
            return raw.squeeze(-1)
        raw = self.connector(self.encode_input(z_tilde, a_z))
        if self.readout_head is not None:
            raw = self.readout_head(raw)
        return raw
