"""Load a QUBO coefficient matrix from the shared Examples directory."""

import os

import numpy as np


def load_QUBO(num_qubits: int) -> np.matrix:
    """Load the QUBO matrix Q for a given problem size.

    Expects a whitespace-delimited file at ``../Examples/QUBO_{num_qubits}.txt``
    containing num_qubits * num_qubits numerical values (row-major).
    """
    file_path = os.path.join("..", "Examples", f"QUBO_{num_qubits}.txt")
    with open(file_path, "r") as file:
        Q = np.loadtxt(file)

    Q = np.asmatrix(Q)
    dim = int(np.size(Q) ** 0.5)
    Q = np.reshape(Q, (dim, dim))
    return Q
