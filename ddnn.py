"""Classical neural-network baseline for higher-order energy minimization.

Trains a small feedforward network to directly output a (relaxed, sigmoid)
binary design vector minimizing a higher-order energy loss — a fully
classical differentiable relaxation of the same problem VQOA solves with a
quantum circuit ansatz, used as a baseline comparison.

Shares the same "N independent restarts, keep the global best" MPI pattern
as VQOA via MPIParallelOptimizer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from interactions import Interaction, higher_order_energy
from optimizer import MPIParallelOptimizer


class QUBONetwork(nn.Module):
    """Small MLP mapping a random input vector to a relaxed binary design vector."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # per-variable probability of being 1


class DDNN(MPIParallelOptimizer):
    """Trains QUBONetwork to minimize a higher-order energy loss.

    Args:
        num_qubits: Number of design variables (network input/output width).
        interactions: Higher-order interaction terms, see interactions.py.
        epochs: Number of gradient-descent steps.
        learning_rate: Adam learning rate.
        use_mpi: Run as independent MPI restarts (see MPIParallelOptimizer).
    """

    def __init__(
        self,
        num_qubits: int,
        interactions: list[Interaction],
        epochs: int = 500,
        learning_rate: float = 0.001,
        use_mpi: bool = False,
    ):
        super().__init__(use_mpi=use_mpi)
        self.num_qubits = num_qubits
        self.interactions = interactions
        self.epochs = epochs
        self.learning_rate = learning_rate

    @property
    def design_length(self) -> int:
        return self.num_qubits

    def __repr__(self) -> str:
        return f"DDNN(num_qubits={self.num_qubits}, epochs={self.epochs}, lr={self.learning_rate})"

    def _run_once(self) -> tuple[float, np.ndarray, list[float]]:
        model = QUBONetwork(self.num_qubits)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        data = torch.rand(1, self.num_qubits)

        best_loss = float("inf")
        best_design = None
        loss_trace: list[float] = []

        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = higher_order_energy(output[0], self.interactions)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_trace.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_design = (output > 0.5).float()[0].detach().numpy()

        return best_loss, best_design, loss_trace
