# DVQOA — Distributed Variational Quantum Optimization Algorithm

Circuit-cutting variational quantum optimization for combinatorial design
problems (QUBO, higher-order polynomial energies, and physical inverse-design
problems such as transparent radiative coolers and optical-diode gratings),
with an optional classical neural-network baseline and optional MPI
parallelism.

## How it works

A `num_qubits`-qubit design problem is split into `num_cutting` independent
sub-circuits, each a shallow hardware-efficient ansatz (Hadamard + repeated
parameterized `Ry` layers). Each sub-circuit's output is turned into one or
more design digits by a **readout strategy**:

- **Binary** (`SamplingReadout`) — measure the sub-circuit and keep the most
  likely bitstring. Any number of qubits per sub-circuit.
- **N-ary** (`LabelStateReadout`) — classify a single qubit's final
  statevector by nearest reference ("label") state, giving a ternary,
  quaternary, etc. digit. Requires exactly one qubit per sub-circuit.

The assembled design vector is scored by a cost function (QUBO energy,
higher-order polynomial energy, or a physics simulation), and
[COBYLA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)
tunes the ansatz parameters to minimize it. Everything above is implemented
once, in the `VQOA` class — running serially, running many independent
restarts under MPI, and switching between binary/ternary/quaternary
encoding are all just constructor arguments, not separate scripts.

A classical baseline, `DDNN`, trains a small feedforward network to directly
output a relaxed design vector minimizing the same kind of energy loss, for
comparison against the quantum approach.

## Repository layout

```
optimizer.py    MPIParallelOptimizer — shared "run N independent restarts,
                keep the global best" base class. mpi4py is only imported
                lazily, inside this module, so serial use needs no MPI
                install at all.
interactions.py Loading/evaluating higher-order polynomial interaction
                terms (shared by the higher-order, n-ary, and DDNN cost
                functions).
vqoa.py         The VQOA optimizer class, plus the SamplingReadout /
                LabelStateReadout readout strategies and the TERNARY_LABELS
                / QUATERNARY_LABELS reference states.
ddnn.py         The DDNN classical-NN baseline class.

load_QUBO.py                   QUBO matrix loader.
TMM_calculation.py             Transparent radiative cooler FOM (2-bit /
                                4-material layer encoding), via tmm_fast.
TMM_calculation_three_states.py Same, but 1-trit / 3-material encoding.
OD_calculation.py              Optical-diode grating FOM (RCWA), via meent.

run_qubo.py          Minimize a QUBO / TMM / RCWA cost (--cost qubo|tmm|rcwa).
run_higher_order.py  Minimize a higher-order polynomial energy, binary encoding.
run_n_ary.py         Same, but n-ary encoding (--states 3|4).
run_trc.py           Radiative cooler design, ternary encoding.
run_ddnn.py          Classical DDNN baseline.

legacy/         Original, single-purpose per-problem scripts, kept for
                reference. Not maintained — use the run_*.py scripts above.
```

## Requirements

- Python 3.9+
- `numpy`, `scipy`
- `qiskit`, `qiskit-aer` (all `run_*.py` scripts except `run_ddnn.py`)
- `pandas`, [`tmm_fast`](https://github.com/MLResearchAtOSRAM/tmm_fast) (`run_qubo.py --cost tmm`, `run_trc.py`)
- [`meent`](https://github.com/kc-ml2/meent) (`run_qubo.py --cost rcwa`)
- `torch` (`run_ddnn.py`)
- `mpi4py` — **optional**, only needed when passing `--mpi`

```bash
pip install numpy scipy qiskit qiskit-aer pandas torch mpi4py
pip install tmm_fast meent  # only if you need those specific cost functions
```

## Input data

The cost functions expect a sibling `Examples/` directory (i.e. `../Examples`
relative to this repo) containing:

- `QUBO_{N}.txt` — QUBO matrix for an N-qubit problem (`run_qubo.py --cost qubo`)
- `Size{N}_Order{M}.txt` — higher-order interaction terms, one
  `var_1,...,var_k,coefficient` line per term, 1-based indices
  (`run_higher_order.py`, `run_n_ary.py`, `run_ddnn.py`)
- `solar_spectrum.txt`, `dielectric_ref.txt` — TMM material/target data
  (`run_qubo.py --cost tmm`, `run_trc.py`)

Check which `N`/`M` a given run script expects at the top of that script
(e.g. `run_n_ary.py --states 4` expects `Size14_Order2.txt`) before running
it, and generate/add the corresponding file if it's missing.

## Usage

Serial:

```bash
python run_qubo.py --cost qubo
python run_higher_order.py
python run_n_ary.py --states 3
python run_trc.py
python run_ddnn.py
```

MPI (rank 0 aggregates; ranks 1..N-1 each run one independent restart, so
launch with at least 2 processes):

```bash
mpiexec -n 5 python run_qubo.py --cost qubo --mpi
mpiexec -n 5 python run_n_ary.py --states 4 --mpi --save-prefix results/n_ary_4
```

`--save-prefix` (MPI mode only) writes the aggregated best result to
`<prefix>.txt` and the winning worker's full loss trace to
`<prefix>_lossdata.txt`.

## Using VQOA programmatically

```python
from vqoa import VQOA, LabelStateReadout, TERNARY_LABELS
from interactions import load_interactions, higher_order_energy

interactions = load_interactions(num_qubits=10, max_order=3)

vqoa = VQOA(
    num_qubits=10,
    cost_function=lambda design: higher_order_energy(design, interactions),
    readout=LabelStateReadout(TERNARY_LABELS),  # omit for binary sampling
    num_layers=3,
    num_repeats=3,
    max_iter=5000,
    use_mpi=False,  # set True + run under mpiexec to parallelize restarts
)
result = vqoa.run()
print(result.loss, result.design)
```
