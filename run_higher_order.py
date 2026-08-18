#!/usr/bin/env python3
"""Run VQOA to minimize a higher-order polynomial energy via circuit-cutting.

Serial:  python run_higher_order.py
MPI:     mpiexec -n 5 python run_higher_order.py --mpi
"""

import argparse
import warnings

from vqoa import VQOA
from interactions import load_interactions, higher_order_energy

warnings.simplefilter("ignore")

NUM_QUBITS = 26
MAX_ORDER = 3
NUM_LAYERS = 3
NUM_REPEATS = 3
MAX_ITER = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpi", action="store_true",
                         help="Run as independent MPI restarts (needs mpiexec -n >= 2).")
    parser.add_argument("--save-prefix", default=None,
                         help="MPI mode: save best result / loss trace to <prefix>.txt / <prefix>_lossdata.txt")
    args = parser.parse_args()

    interactions = load_interactions(NUM_QUBITS, MAX_ORDER)

    vqoa = VQOA(
        num_qubits=NUM_QUBITS,
        cost_function=lambda design: higher_order_energy(design, interactions),
        num_layers=NUM_LAYERS,
        num_repeats=NUM_REPEATS,
        max_iter=MAX_ITER,
        use_mpi=args.mpi,
        verbose=not args.mpi,
    )
    result = vqoa.run(save_prefix=args.save_prefix)

    if result is not None:
        print(f"best loss: {result.loss:.6f}")
        print(f"best design: {result.design}")
        print(f"elapsed time: {result.elapsed_time:.6f}s")


if __name__ == "__main__":
    main()
