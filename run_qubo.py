#!/usr/bin/env python3
"""Run VQOA to minimize a QUBO / radiative-cooler / optical-diode design cost.

Serial:  python run_qubo.py --cost qubo
MPI:     mpiexec -n 5 python run_qubo.py --cost qubo --mpi
"""

import argparse
import warnings

from vqoa import VQOA
from load_QUBO import load_QUBO
from TMM_calculation import cal_FOM_TMM
from OD_calculation import cal_FOM_RCWA

warnings.simplefilter("ignore")

NUM_QUBITS = 30
NUM_LAYERS = 7
NUM_REPEATS = 7
MAX_ITER = 5000
RCWA_WAVELENGTH = 800  # nm; only used for --cost rcwa


def make_cost_function(name: str):
    if name == "qubo":
        Q = load_QUBO(NUM_QUBITS)
        return lambda design: float(design @ Q @ design.T)
    if name == "tmm":
        return lambda design: float(cal_FOM_TMM(design)[0][0])
    if name == "rcwa":
        return lambda design: float(cal_FOM_RCWA(design, RCWA_WAVELENGTH))
    raise ValueError(f"Unknown cost function: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost", choices=["qubo", "tmm", "rcwa"], default="qubo",
                         help="Design problem to optimize (default: qubo).")
    parser.add_argument("--mpi", action="store_true",
                         help="Run as independent MPI restarts (needs mpiexec -n >= 2).")
    parser.add_argument("--save-prefix", default=None,
                         help="MPI mode: save best result / loss trace to <prefix>.txt / <prefix>_lossdata.txt")
    args = parser.parse_args()

    vqoa = VQOA(
        num_qubits=NUM_QUBITS,
        cost_function=make_cost_function(args.cost),
        num_layers=NUM_LAYERS,
        num_repeats=NUM_REPEATS,
        max_iter=MAX_ITER,
        use_mpi=args.mpi,
        verbose=not args.mpi,
    )
    result = vqoa.run(save_prefix=args.save_prefix)

    if result is not None:
        print(f"cost function: {args.cost}")
        print(f"best loss: {result.loss:.6f}")
        print(f"best design: {result.design}")
        print(f"elapsed time: {result.elapsed_time:.6f}s")


if __name__ == "__main__":
    main()
