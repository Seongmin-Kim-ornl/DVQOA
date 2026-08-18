#!/usr/bin/env python3
"""Run VQOA to minimize a QUBO / radiative-cooler / optical-diode design cost.

Serial:  python run_qubo.py --cost qubo --num-qubits 30
MPI:     mpiexec -n 5 python run_qubo.py --cost qubo --num-qubits 30 --mpi
"""

import argparse
import warnings

from vqoa import VQOA
from load_QUBO import load_QUBO

warnings.simplefilter("ignore")

NUM_QUBITS = 30  # --cost qubo needs a matching ../Examples/QUBO_{N}.txt
NUM_LAYERS = 7
NUM_REPEATS = 7
MAX_ITER = 5000
RCWA_WAVELENGTH = 800  # nm; only used for --cost rcwa


def make_cost_function(name: str, num_qubits: int):
    if name == "qubo":
        Q = load_QUBO(num_qubits)
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
    parser.add_argument("--num-qubits", type=int, default=NUM_QUBITS,
                         help=f"Problem size / number of design variables (default: {NUM_QUBITS}). "
                              "For --cost qubo, must match an existing ../Examples/QUBO_{N}.txt.")
    parser.add_argument("--num-cutting", type=int, default=None,
                         help="Number of independent sub-circuits (default: num_qubits // 10, "
                              "min 1). Must evenly divide --num-qubits; override this if it "
                              "doesn't for your chosen problem size.")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS, help=f"default: {NUM_LAYERS}")
    parser.add_argument("--num-repeats", type=int, default=NUM_REPEATS, help=f"default: {NUM_REPEATS}")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER, help=f"default: {MAX_ITER}")
    parser.add_argument("--mpi", action="store_true",
                         help="Run as independent MPI restarts (needs mpiexec -n >= 2).")
    parser.add_argument("--save-prefix", default=None,
                         help="MPI mode: save best result / loss trace to <prefix>.txt / <prefix>_lossdata.txt")
    args = parser.parse_args()

    vqoa = VQOA(
        num_qubits=args.num_qubits,
        cost_function=make_cost_function(args.cost, args.num_qubits),
        num_cutting=args.num_cutting,
        num_layers=args.num_layers,
        num_repeats=args.num_repeats,
        max_iter=args.max_iter,
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
