"""Loading and evaluating higher-order (polynomial) interaction terms.

Interaction files (``../Examples/Size{N}_Order{M}.txt``) list one term per
line as ``var_1,...,var_k,coefficient``, using 1-based variable indices.
The energy of a design vector is the sum of coefficient * product(digits)
over all terms.
"""

from __future__ import annotations

import os

Interaction = tuple[list[int], float]

EXAMPLES_DIR = os.path.join("..", "Examples")


def load_interactions(num_qubits: int, max_order: int, examples_dir: str = EXAMPLES_DIR) -> list[Interaction]:
    """Load higher-order interaction terms for a given problem size/order."""
    filename = os.path.join(examples_dir, f"Size{num_qubits}_Order{max_order}.txt")
    interactions: list[Interaction] = []
    with open(filename, "r") as f:
        for line in f:
            *variables, coefficient = line.strip().split(",")
            interactions.append(([int(v) for v in variables], float(coefficient)))
    return interactions


def higher_order_energy(design_vector, interactions: list[Interaction]):
    """Sum coefficient * product(design_vector[v - 1] for v in variables) over all terms.

    Works with plain numpy/float values as well as autograd-tracked
    tensors (e.g. torch), since it only ever uses +, *, and indexing.
    """
    energy = 0.0
    for variables, coefficient in interactions:
        product = 1
        for var in variables:
            product = product * design_vector[var - 1]  # variable indices are 1-based
        energy = energy + coefficient * product
    return energy
