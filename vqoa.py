"""Variational Quantum Optimization Algorithm (VQOA) via circuit-cutting.

A `num_qubits`-qubit design problem is split into `num_cutting` independent
sub-circuits, each a shallow hardware-efficient ansatz (H + repeated Ry
layers). A pluggable *readout* strategy turns each sub-circuit's final
quantum state into one or more design digits:

- :class:`SamplingReadout` — measures the sub-circuit and keeps the most
  likely bitstring (binary digits, any number of qubits per sub-circuit).
- :class:`LabelStateReadout` — classifies a single qubit's statevector by
  nearest reference state (n-ary digits, exactly one qubit per sub-circuit).

The assembled design vector is scored by a user-supplied cost function, and
COBYLA tunes the ansatz parameters to minimize it.

This one class replaces what used to be six near-duplicate scripts
(VQOA.py, VQOA_higher_order.py, VQOA_N-ary.py, VQOA_TRC_multistates.py,
DVQOA*.py, PQC_MPI_4vector.py): the binary/MPI/n-ary variants are now just
different constructor arguments. See the run_*.py scripts for examples.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer, AerSimulator
from qiskit.primitives import BackendSampler
from scipy.optimize import minimize

from optimizer import MPIParallelOptimizer

CostFunction = Callable[[np.ndarray], float]

# Reference single-qubit states for n-ary label-state readout, matching the
# exact vectors used in the original per-encoding research scripts.
TERNARY_LABELS = [
    np.array([1, 0], dtype=complex),               # digit 0: |0>
    np.array([-0.5, -0.8660], dtype=complex),       # digit 1
    np.array([-0.5, 0.8660], dtype=complex),        # digit 2
]
QUATERNARY_LABELS = [
    np.array([1, 0], dtype=complex),   # digit 0
    np.array([0, -1], dtype=complex),  # digit 1
    np.array([-1, 0], dtype=complex),  # digit 2
    np.array([0, 1], dtype=complex),   # digit 3
]


def int_to_padded_binary_array(num: int, num_bits: int) -> np.ndarray:
    """Convert a non-negative integer to a zero-padded binary digit array."""
    if num < 0:
        raise ValueError("We cannot process negative numbers.")
    binary_string = bin(num)[2:].zfill(num_bits)
    return np.array(list(binary_string), dtype=np.float32)


def _build_ansatz_circuit(num_cut_qubits: int, sub_theta: np.ndarray, num_layers: int,
                           num_repeats: int, measure: bool) -> QuantumCircuit:
    """H on every qubit, then the same Ry layer block applied num_repeats times."""
    circuit = QuantumCircuit(num_cut_qubits)
    circuit.h(range(num_cut_qubits))
    for _ in range(num_repeats):
        for layer in range(num_layers):
            for qubit in range(num_cut_qubits):
                circuit.ry(sub_theta[layer * num_cut_qubits + qubit], qubit)
    if measure:
        circuit.measure_all()
    return circuit


class SamplingReadout:
    """Binary readout: sample the sub-circuit and keep the most likely bitstring."""

    needs_measurement = True
    requires_single_qubit_cuts = False

    def __init__(self, shots: int = 2000):
        self.shots = shots
        self._backend = BackendSampler(
            backend=AerSimulator(method="automatic"),
            skip_transpilation=False,
            options={"shots": shots},
        )

    def __call__(self, circuit: QuantumCircuit, num_cut_qubits: int) -> np.ndarray:
        result = self._backend.run(circuit, shots=self.shots).result()
        counts = result.quasi_dists[0]
        most_likely_bits, _ = max(counts.items(), key=lambda item: item[1])
        return int_to_padded_binary_array(most_likely_bits, num_cut_qubits)

    def __repr__(self) -> str:
        return f"SamplingReadout(shots={self.shots})"


class LabelStateReadout:
    """N-ary readout: classify a single qubit's statevector by nearest label state.

    Requires exactly one qubit per sub-circuit (num_cutting == num_qubits).
    """

    needs_measurement = False
    requires_single_qubit_cuts = True

    def __init__(self, label_states: list[np.ndarray]):
        if len(label_states) < 2:
            raise ValueError("Need at least 2 label states.")
        self.label_states = [np.asarray(state, dtype=complex) for state in label_states]
        self._backend = Aer.get_backend("statevector_simulator")

    def __call__(self, circuit: QuantumCircuit, num_cut_qubits: int) -> np.ndarray:
        if num_cut_qubits != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one qubit per sub-circuit, "
                f"got num_cut_qubits={num_cut_qubits}."
            )
        result = self._backend.run(circuit, shots=1).result()
        state = result.get_statevector(circuit)
        state = np.asarray(state) / np.linalg.norm(state)
        distances = [np.linalg.norm(state - label) for label in self.label_states]
        digit = int(np.argmin(distances))
        return np.array([digit], dtype=np.float32)

    def __repr__(self) -> str:
        return f"LabelStateReadout(num_states={len(self.label_states)})"


class VQOA(MPIParallelOptimizer):
    """Circuit-cutting variational optimizer over a binary or n-ary design vector.

    Args:
        num_qubits: Total number of design variables.
        cost_function: Scores a fully assembled design vector (lower is better).
        readout: SamplingReadout() (default, binary) or a LabelStateReadout
            (n-ary; requires num_cutting == num_qubits).
        num_cutting: Number of independent sub-circuits. Defaults to
            num_qubits // 10 for SamplingReadout, or num_qubits for
            LabelStateReadout. Must evenly divide num_qubits.
        num_layers: Distinct parameterized Ry values per sub-circuit qubit.
        num_repeats: Number of times the Ry layer block is reapplied
            (Ry rotations compose additively on the same qubit, so this
            scales the effective rotation rather than adding new degrees
            of freedom).
        max_iter: Maximum COBYLA iterations.
        use_mpi: Run as independent MPI restarts (see MPIParallelOptimizer).
        verbose: Print progress every 50 iterations (serial mode only).
    """

    def __init__(
        self,
        num_qubits: int,
        cost_function: CostFunction,
        readout: SamplingReadout | LabelStateReadout | None = None,
        num_cutting: int | None = None,
        num_layers: int = 3,
        num_repeats: int = 3,
        max_iter: int = 5000,
        use_mpi: bool = False,
        verbose: bool = False,
    ):
        super().__init__(use_mpi=use_mpi)
        self.num_qubits = num_qubits
        self.cost_function = cost_function
        self.readout = readout or SamplingReadout()
        self.num_layers = num_layers
        self.num_repeats = num_repeats
        self.max_iter = max_iter
        self.verbose = verbose

        default_cutting = num_qubits if self.readout.requires_single_qubit_cuts else max(1, num_qubits // 10)
        self.num_cutting = num_cutting or default_cutting
        if num_qubits % self.num_cutting != 0:
            raise ValueError(f"num_qubits ({num_qubits}) must be divisible by num_cutting ({self.num_cutting}).")
        self.num_cut_qubits = num_qubits // self.num_cutting
        if self.readout.requires_single_qubit_cuts and self.num_cut_qubits != 1:
            raise ValueError(
                f"{type(self.readout).__name__} requires one qubit per sub-circuit; "
                f"got num_cut_qubits={self.num_cut_qubits}. Set num_cutting={num_qubits}."
            )

    @property
    def design_length(self) -> int:
        return self.num_qubits

    def __repr__(self) -> str:
        return (
            f"VQOA(num_qubits={self.num_qubits}, num_cutting={self.num_cutting}, "
            f"num_cut_qubits={self.num_cut_qubits}, num_layers={self.num_layers}, "
            f"num_repeats={self.num_repeats}, readout={self.readout!r})"
        )

    def _objective(self, params: np.ndarray) -> float:
        design = np.zeros(self.num_qubits)

        for cut in range(self.num_cutting):
            offset = cut * self.num_cut_qubits * self.num_layers
            sub_theta = params[offset:offset + self.num_layers * self.num_cut_qubits]

            circuit = _build_ansatz_circuit(
                self.num_cut_qubits, sub_theta, self.num_layers, self.num_repeats,
                measure=self.readout.needs_measurement,
            )
            digits = self.readout(circuit, self.num_cut_qubits)
            design[cut * self.num_cut_qubits:(cut + 1) * self.num_cut_qubits] = digits

        loss = self.cost_function(design)
        self._loss_trace.append(loss)

        if loss < self._best_loss:
            self._best_loss = loss
            self._best_design = design

        if self.verbose and len(self._loss_trace) % 50 == 0:
            print(f"iter {len(self._loss_trace)}: loss={loss:.6f}, best={self._best_loss:.6f}")

        return loss

    def _run_once(self) -> tuple[float, np.ndarray, list[float]]:
        self._best_loss = float("inf")
        self._best_design = None
        self._loss_trace: list[float] = []

        initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, self.num_layers * self.num_qubits)
        minimize(self._objective, initial_params, method="COBYLA", options={"maxiter": self.max_iter})

        return self._best_loss, self._best_design, self._loss_trace
