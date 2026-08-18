"""Figure-of-merit (FOM) calculation for a transparent radiative cooler stack.

Uses `tmm_fast` (vectorized transfer-matrix method) to simulate a multilayer
dielectric stack and score it against an ideal transmission spectrum (a
bandpass window in the visible range).

Design encoding: each layer is chosen from 4 materials using 2 bits.
"""

import os

import numpy as np
import pandas as pd
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm

EXAMPLES_DIR = os.path.join("..", "Examples")


def cal_FOM_TMM(bit_string: np.ndarray) -> np.ndarray:
    """Compute the transmission-matching FOM for a 2-bit-per-layer design.

    Args:
        bit_string: Binary design vector; each pair of bits selects one of
            4 dielectric materials for a layer (00=SiO2, 01=Si3N4,
            10=Al2O3, 11=TiO2).

    Returns:
        A (1, 1) array holding the FOM (lower is better: closer match to
        the ideal transmission spectrum).
    """
    qv_ii = bit_string
    num_qubits = np.size(qv_ii)

    # Simulation parameters (wavelength, incident angle, layer thickness)
    wl = np.linspace(300, 2500, 2201) * 1e-9  # wavelength, meters
    theta = np.linspace(0, 0, 1) * (np.pi / 180)  # incident angle
    num_layers = int(num_qubits / 2)
    layer_thickness = 50  # nm
    num_stacks = 1
    total_layers = num_layers + 2
    upper_medium = 1.4  # e.g. PDMS
    lower_medium = 1.45  # substrate, SiO2

    # Ideal target transmission: passband over 400-750 nm
    solar_spectrum = pd.read_csv(os.path.join(EXAMPLES_DIR, "solar_spectrum.txt"), sep=",", header=None)
    solar_spectrum = solar_spectrum.values.astype(np.float32)
    ideal_T = np.zeros((1, np.size(wl)))
    ideal_T[0, 0:400 - 300] = 0
    ideal_T[0, 400 - 300:750 - 300 + 1] = 1
    ideal_T[0, 750 - 300 + 1:] = 0
    ideal_trans_energy = np.multiply(ideal_T, solar_spectrum)

    # Layer thicknesses (semi-infinite upper/lower media)
    thickness = np.ones((1, total_layers))
    thickness[:, 1:] = layer_thickness
    thickness[:, 0] = np.inf
    thickness[:, -1] = np.inf
    thickness = thickness * 1e-9

    # Refractive index lookup: 0=SiO2, 1=Si3N4, 2=Al2O3, 3=TiO2
    dielectric_ref = pd.read_csv(os.path.join(EXAMPLES_DIR, "dielectric_ref.txt"), sep=",", header=None)
    dielectric_ref = dielectric_ref.values.astype(np.float32)

    mater_index = np.zeros((num_layers + 1, np.size(wl)))  # index 0 is the upper medium
    for nl in range(num_layers):
        bit_pair = (int(qv_ii[nl * 2]), int(qv_ii[nl * 2 + 1]))
        material = {
            (0, 0): 0,  # SiO2
            (0, 1): 1,  # Si3N4
            (1, 0): 2,  # Al2O3
            (1, 1): 3,  # TiO2
        }[bit_pair]
        mater_index[nl + 1] = dielectric_ref[material, :]

    # Assemble the full stack (upper medium / layers / substrate)
    M = np.ones((num_stacks, total_layers, wl.shape[0]))
    M[:, 0] = upper_medium
    M[:, 1:-1] = mater_index[1:]
    M[:, -1] = lower_medium

    results = tmm("p", M, thickness, theta, wl, device="cpu", timer=True)
    T = results[0]["T"]

    trans_energy = np.multiply(T[0], solar_spectrum)
    FOM_cal = np.sum(np.square(np.subtract(trans_energy, ideal_trans_energy)))
    FOM = FOM_cal / np.sum(np.square(solar_spectrum)) * 10
    return FOM.reshape(1, 1)
