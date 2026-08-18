"""Shared "N independent restarts, keep the global best" optimizer base class.

Both VQOA (quantum circuit ansatz, see vqoa.py) and DDNN (classical NN
baseline, see ddnn.py) follow the same execution pattern: run one full
optimization, or — when MPI is enabled — run several independent copies in
parallel and keep whichever found the lowest loss. This module factors that
pattern out so it is implemented (and tested) exactly once.

mpi4py is only imported lazily, inside ``_run_mpi``, so serial use never
requires it to be installed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationResult:
    """Outcome of an optimizer run."""

    loss: float
    design: np.ndarray
    elapsed_time: float
    # Populated only for the rank-0 result of an MPI run:
    all_losses: np.ndarray | None = None
    all_designs: np.ndarray | None = None


class MPIParallelOptimizer(ABC):
    """Base class for optimizers that can run standalone or as MPI restarts.

    Subclasses implement :meth:`_run_once`, a single independent
    optimization run, and the :attr:`design_length` property. Everything
    about *how* that single run gets parallelized across MPI ranks and
    aggregated lives here.
    """

    def __init__(self, use_mpi: bool = False):
        self.use_mpi = use_mpi

    @property
    @abstractmethod
    def design_length(self) -> int:
        """Length of the design vector returned by _run_once (e.g. num_qubits)."""

    @abstractmethod
    def _run_once(self) -> tuple[float, np.ndarray, list[float]]:
        """Run one full optimization.

        Returns:
            (best_loss, best_design, loss_trace)
        """

    def run(self, save_prefix: str | None = None) -> OptimizationResult | None:
        """Execute the optimizer.

        Args:
            save_prefix: MPI mode only. If given, rank 0 writes the
                aggregated result to ``{save_prefix}.txt`` and the winning
                worker writes its full loss trace to
                ``{save_prefix}_lossdata.txt``.

        Returns:
            The result in serial mode. In MPI mode, only rank 0 gets the
            aggregated :class:`OptimizationResult`; every other rank
            returns ``None`` (it only contributed a restart).
        """
        if not self.use_mpi:
            start = time.time()
            loss, design, _ = self._run_once()
            return OptimizationResult(loss=loss, design=design, elapsed_time=time.time() - start)
        return self._run_mpi(save_prefix)

    def _run_mpi(self, save_prefix: str | None) -> OptimizationResult | None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        num_workers = size - 1  # rank 0 is an idle aggregator

        if num_workers < 1:
            raise RuntimeError(
                "use_mpi=True needs at least 2 MPI processes (rank 0 aggregates, "
                "ranks 1..N-1 each run one restart). Launch with: "
                "mpiexec -n <workers + 1> python <script>.py --mpi"
            )

        if rank == 0:
            print(f"[MPI] {self!r}")
            print(f"[MPI] {num_workers} worker(s) starting...")

        start_time = MPI.Wtime()
        loss_trace: list[float] = []

        if rank != 0:
            loss, design, loss_trace = self._run_once()
            comm.send(loss, dest=0)
            comm.send(design, dest=0)

        result = None
        min_arg = None

        if rank == 0:
            all_losses = np.zeros(num_workers)
            all_designs = np.zeros((num_workers, self.design_length))
            for source_rank in range(1, size):
                all_losses[source_rank - 1] = comm.recv(source=source_rank)
                all_designs[source_rank - 1] = comm.recv(source=source_rank)

            min_arg = int(np.argmin(all_losses))
            best_loss = all_losses[min_arg]
            best_design = all_designs[min_arg, :]
            elapsed = MPI.Wtime() - start_time

            print(f"[MPI] done in {elapsed:.3f}s")
            print(f"[MPI] worker losses: {all_losses}")
            print(f"[MPI] best loss: {best_loss:.6f} (worker rank {min_arg + 1})")

            if save_prefix:
                with open(f"{save_prefix}.txt", "w") as f:
                    f.write(f"{elapsed}\n{best_loss}\n{best_design}\n")
                print(f"[MPI] result saved to {save_prefix}.txt")
                for worker_rank in range(1, size):
                    comm.send(min_arg, dest=worker_rank)

            result = OptimizationResult(
                loss=best_loss, design=best_design, elapsed_time=elapsed,
                all_losses=all_losses, all_designs=all_designs,
            )

        if rank != 0 and save_prefix:
            min_arg = comm.recv(source=0)
            if min_arg + 1 == rank:
                loss_path = f"{save_prefix}_lossdata.txt"
                with open(loss_path, "w") as f:
                    f.write(",".join(f"{v:.6f}" for v in loss_trace) + "\n")
                print(f"[MPI] rank {rank}: loss trace saved to {loss_path}")

        return result
