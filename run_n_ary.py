#!/usr/bin/env python3
"""Run VQOA with n-ary (ternary/quaternary) design-variable encoding.

Each design variable is read out by classifying a single qubit's final
statevector against a set of reference "label" states, instead of binary
measurement sampling. This one script, selected via --states, replaces
what used to be three separate near-duplicate scripts: VQOA_N-ary.py,
DVQOA_N-ary.py, and PQC_MPI_4vector.py.

Serial:  python run_n_ary.py --states 3
MPI:     mpiexec -n 5 python run_n_ary.py --states 4 --mpi
"""

import argparse
import warnings

from vqoa import VQOA, LabelStateReadout, TERNARY_LABELS, QUATERNARY_LABELS
from interactions import load_interactions, higher_order_energy

warnings.simplefilter("ignore")

# Presets matching the original per-encoding problem instances.
PRESETS = {
    3: dict(num_qubits=10, max_order=3, label_states=TERNARY_LABELS),
    4: dict(num_qubits=14, max_order=2, label_states=QUATERNARY_LABELS),
}
NUM_LAYERS = 3
NUM_REPEATS = 3
MAX_ITER = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=int, choices=sorted(PRESETS), default=3,
                         help="Number of digit values per design variable (default: 3).")
    parser.add_argument("--mpi", action="store_true",
                         help="Run as independent MPI restarts (needs mpiexec -n >= 2).")
    parser.add_argument("--save-prefix", default=None,
                         help="MPI mode: save best result / loss trace to <prefix>.txt / <prefix>_lossdata.txt")
    args = parser.parse_args()

    preset = PRESETS[args.states]
    interactions = load_interactions(preset["num_qubits"], preset["max_order"])

    vqoa = VQOA(
        num_qubits=preset["num_qubits"],
        cost_function=lambda design: higher_order_energy(design, interactions),
        readout=LabelStateReadout(preset["label_states"]),
        num_layers=NUM_LAYERS,
        num_repeats=NUM_REPEATS,
        max_iter=MAX_ITER,
        use_mpi=args.mpi,
        verbose=not args.mpi,
    )
    result = vqoa.run(save_prefix=args.save_prefix)

    if result is not None:
        print(f"states: {args.states}")
        print(f"best loss: {result.loss:.6f}")
        print(f"best design: {result.design}")
        print(f"elapsed time: {result.elapsed_time:.6f}s")


if __name__ == "__main__":
    main()
