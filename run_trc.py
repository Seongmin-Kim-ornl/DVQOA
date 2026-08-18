#!/usr/bin/env python3
"""Run VQOA for transparent radiative cooler design (ternary material encoding).

Serial:  python run_trc.py
MPI:     mpiexec -n 5 python run_trc.py --mpi
"""

import argparse
import warnings

from vqoa import VQOA, LabelStateReadout, TERNARY_LABELS
from TMM_calculation_three_states import cal_FOM_TMM_three_states

warnings.simplefilter("ignore")

NUM_QUBITS = 20
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

    vqoa = VQOA(
        num_qubits=NUM_QUBITS,
        cost_function=lambda design: cal_FOM_TMM_three_states(design)[0][0],
        readout=LabelStateReadout(TERNARY_LABELS),
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
