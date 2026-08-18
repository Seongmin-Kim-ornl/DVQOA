#!/usr/bin/env python3
"""Run the classical DDNN baseline for higher-order energy minimization.

Serial:  python run_ddnn.py
MPI:     mpiexec -n 5 python run_ddnn.py --mpi
"""

import argparse
import warnings

from ddnn import DDNN
from interactions import load_interactions

warnings.simplefilter("ignore")

NUM_QUBITS = 26
MAX_ORDER = 3
EPOCHS = 500
LEARNING_RATE = 0.001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpi", action="store_true",
                         help="Run as independent MPI restarts (needs mpiexec -n >= 2).")
    args = parser.parse_args()

    interactions = load_interactions(NUM_QUBITS, MAX_ORDER)
    model = DDNN(
        num_qubits=NUM_QUBITS,
        interactions=interactions,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        use_mpi=args.mpi,
    )
    result = model.run()

    if result is not None:
        print(f"best loss: {result.loss:.6f}")
        print(f"best design: {result.design}")
        print(f"elapsed time: {result.elapsed_time:.6f}s")


if __name__ == "__main__":
    main()
