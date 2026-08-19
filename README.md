# DVQOA — Distributed Variational Quantum Optimization Algorithm

Circuit-cutting variational quantum optimization for QUBO, and materials design problems,
with optional MPI parallelism.

`run_qubo.py` is the entry point for the cost functions.

## How it works

A `num_qubits`-qubit design problem is split into `num_cutting` independent
sub-circuits, each a shallow hardware-efficient ansatz (Hadamard + repeated
parameterized `Ry` layers). Each sub-circuit is measured and its most likely
bitstring becomes a slice of the design vector. The assembled design vector
is scored by the chosen cost function (QUBO energy), and
[COBYLA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)
tunes the ansatz parameters to minimize it. This is all implemented once, in
the `VQOA` class (`vqoa.py`) — `run_qubo.py` just configures and calls it.

## Files needed for `run_qubo.py`

```
run_qubo.py         Entry point / CLI.
vqoa.py              The VQOA optimizer class.
optimizer.py         MPIParallelOptimizer — shared "run N independent
                     restarts, keep the global best" base class. mpi4py is
                     only imported lazily, inside this module, so serial
                     runs need no MPI install at all.
load_QUBO.py         QUBO matrix loader (--cost qubo).
```

## Requirements

- Python 3.9+
- `numpy`, `scipy`
- `qiskit`, `qiskit-aer`
- `pandas`
- `mpi4py` — **optional**, only needed when passing `--mpi`

```bash
pip install numpy scipy qiskit qiskit-aer pandas mpi4py
```

## Input data

Cost functions read from an `Examples/` directory next to the scripts
(resolved relative to your current working directory when you launch
`python run_qubo.py`):

- `QUBO_{N}.txt` — QUBO matrix for an N-qubit problem (`--cost qubo`)


## Usage

Serial:

```bash
python run_qubo.py --cost qubo --num-qubits 20
```

Useful flags (all optional, see `python run_qubo.py --help`):

- `--cost {qubo}` — design problem to optimize (default: `qubo`)
- `--num-qubits N` — problem size; for `--cost qubo` must match an existing `QUBO_{N}.txt`
- `--num-cutting K` — number of sub-circuits; must evenly divide `--num-qubits`
  (default: `num_qubits // 10`, min 1 — override this if it doesn't divide evenly,
  e.g. `--num-qubits 32 --num-cutting 4`)
- `--num-layers`, `--num-repeats`, `--max-iter` — ansatz/optimizer sizing

MPI (rank 0 aggregates; ranks 1..N-1 each run one independent restart, so
launch with at least 2 processes):

```bash
mpiexec -n 5 python run_qubo.py --cost qubo --num-qubits 20 --mpi
# or, on a Slurm-managed cluster:
srun -n 5 python run_qubo.py --cost qubo --num-qubits 20 --mpi
```

`srun` works as a drop-in replacement for `mpiexec` — `mpi4py` picks up
rank/size from the MPI runtime, not from the launch command. Add
`--save-prefix results/run1` (MPI mode only) to write the aggregated best
result to `results/run1.txt` and the winning worker's full loss trace to
`results/run1_lossdata.txt`.

## Using VQOA programmatically

```python
from vqoa import VQOA
from load_QUBO import load_QUBO

Q = load_QUBO(num_qubits=20)

vqoa = VQOA(
    num_qubits=20,
    cost_function=lambda design: float(design @ Q @ design.T),
    num_layers=5,
    num_repeats=5,
    max_iter=1000,
    use_mpi=False,  # set True + run under mpiexec/srun to parallelize restarts
)
result = vqoa.run()
print(result.loss, result.design)
```

## Citation
`Kim, S., Suh, IS. Advancing scientific discovery and complex optimization through distributed quantum neural networks. npj Comput Mater (2026). https://doi.org/10.1038/s41524-026-02203-w`
