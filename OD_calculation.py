"""Figure-of-merit (FOM) calculation for optical-diode grating design via RCWA.

Uses `meent` (rigorous coupled-wave analysis) to evaluate the forward/backward
transmission asymmetry of a 1D binary grating stack, which is the target
property being optimized for an optical-diode design.
"""

import numpy as np

import meent


def cal_FOM_RCWA(bit_string: np.ndarray, wave: int) -> float:
    """Compute the forward/backward transmission asymmetry FOM for a grating.

    Args:
        bit_string: Binary design vector (0/1 per pixel), split across
            `n_grating_layer` layers.
        wave: Simulation wavelength in nm. Must be one of 600, 800, 1000
            (each has a matching complex refractive index for the metal layer).

    Returns:
        FOM = BT - FT, the difference between backward and forward
        transmission (0 = no diode effect, larger magnitude = stronger
        one-way transmission).
    """
    qv_ii = bit_string
    qv_bits = np.size(qv_ii)  # number of binary design bits

    # Simulation parameters (wavelength, incident angle, layer stack)
    pol = 1  # 0: TE, 1: TM
    n_I = 1  # superstrate refractive index (air)
    n_II = 1.45  # substrate refractive index (SiO2)
    theta = 0 * np.pi / 180
    wavelength = wave
    thickness = [20, 20, 25, 25, 20, 20]
    n_grating_layer = 4
    np_layer = int(qv_bits / n_grating_layer)  # pixels per layer
    period = [wavelength / 4 * 3]
    fourier_order = [40]
    type_complex = np.complex128

    # Refractive indices
    mater_air = 1
    mater_dielectric = 1.45  # SiO2

    if wave == 600:
        mater_metal = 0.1241 - 3.7293j  # Ag at 600 nm
    elif wave == 800:
        mater_metal = 0.1437 - 5.2858j  # Ag at 800 nm
    elif wave == 1000:
        mater_metal = 0.2145 - 6.7612j  # Ag at 1000 nm
    else:
        raise ValueError(f"No refractive index data for wave={wave} nm (expected 600, 800, or 1000)")

    def build_ucell(design_bits: np.ndarray, air_region: slice) -> np.ndarray:
        """Map a binary design vector onto the layer-stack permittivity model."""
        model = np.zeros(qv_bits, dtype="complex_")
        for nl in range(qv_bits):
            if design_bits[nl] == 0:
                model[nl] = mater_air if air_region.start <= nl < air_region.stop else mater_dielectric
            else:
                model[nl] = mater_metal

        ucell = np.zeros((n_grating_layer + 2, 1, np_layer), dtype="complex_")
        for i in range(n_grating_layer):
            for j in range(np_layer):
                if i < 2:
                    ucell[i][0][j] = model[np_layer * i + j]
                else:
                    ucell[2][0][j] = mater_dielectric
                    ucell[3][0][j] = mater_dielectric
                    ucell[i + 2][0][j] = model[np_layer * i + j]
        return ucell

    # Forward calculation: air region is the first layer (index < np_layer)
    ucell_forward = build_ucell(qv_ii, slice(0, np_layer))
    mee_forward = meent.call_mee(
        backend=0, pol=pol, n_top=n_I, n_bot=n_II, theta=theta, fto=fourier_order,
        wavelength=wavelength, period=period, ucell=ucell_forward, thickness=thickness,
        type_complex=type_complex,
    )
    res_forward = mee_forward.conv_solve().res
    FT = res_forward.de_ti.sum()

    # Backward calculation: design is reversed, air region is the last layer
    ucell_backward = build_ucell(np.flip(qv_ii), slice(np_layer * 3, qv_bits))
    mee_backward = meent.call_mee(
        backend=0, pol=pol, n_top=n_II, n_bot=n_I, theta=theta, fto=fourier_order,
        wavelength=wavelength, period=period, ucell=ucell_backward, thickness=thickness,
        type_complex=type_complex,
    )
    res_backward = mee_backward.conv_solve().res
    BT = res_backward.de_ti.sum()

    FOM = BT - FT
    return FOM
