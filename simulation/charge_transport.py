"""
Charge transport in CdTe detector – Hecht equation model.

A photon absorbed at depth x inside the detector generates electron–hole
pairs.  Electrons drift toward the anode (+) and holes toward the cathode (−).
The induced charge on the anode electrode is given by the Hecht equation for
each carrier.

Physical parameters (literature values for room-temperature CdTe)
------------------------------------------------------------------
    μ_e · τ_e  = 3.0 × 10⁻³ cm²/V   (electron mobility-lifetime product)
    μ_h · τ_h  = 4.0 × 10⁻⁴ cm²/V   (hole mobility-lifetime product)
    Detector thickness d  = 2.0 mm
    Bias voltage V_bias   = 600 V  (default, cathode negative)
    E_eh (CdTe)           = 4.43 eV/e-h pair
"""

from __future__ import annotations

import numpy as np

# ─── CdTe material constants ──────────────────────────────────────────────────
MU_TAU_E: float = 3.0e-3   # cm²/V – electron mobility-lifetime product
MU_TAU_H: float = 4.0e-4   # cm²/V – hole mobility-lifetime product
E_EH: float = 4.43          # eV per electron-hole pair
DETECTOR_THICKNESS: float = 0.20  # cm  (2 mm)
BIAS_VOLTAGE: float = 600.0       # V


def hecht_cce(
    x: np.ndarray,
    V_bias: float = BIAS_VOLTAGE,
    d: float = DETECTOR_THICKNESS,
    mu_tau_e: float = MU_TAU_E,
    mu_tau_h: float = MU_TAU_H,
) -> np.ndarray:
    """
    Charge collection efficiency at interaction depth x (from cathode, cm).

    CCE(x) = CCE_e(x) + CCE_h(x)

    where:
        CCE_e = (λ_e/d) · [1 − exp(−(d−x)/λ_e)]   electrons travel x→anode
        CCE_h = (λ_h/d) · [1 − exp(−x/λ_h)]        holes travel x→cathode
        λ_e = μ_e τ_e V/d,   λ_h = μ_h τ_h V/d

    Parameters
    ----------
    x : np.ndarray
        Interaction depth from cathode in **cm** (0 ≤ x ≤ d).
    V_bias : float
        Detector bias voltage in V.
    d : float
        Detector thickness in cm.
    mu_tau_e, mu_tau_h : float
        Mobility-lifetime products (cm²/V).

    Returns
    -------
    cce : np.ndarray
        Charge collection efficiency in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    lam_e = mu_tau_e * V_bias / d   # mean free path of electrons (cm)
    lam_h = mu_tau_h * V_bias / d   # mean free path of holes (cm)

    # Hecht equation per carrier
    cce_e = (lam_e / d) * (1.0 - np.exp(-(d - x) / lam_e))
    cce_h = (lam_h / d) * (1.0 - np.exp(-x / lam_h))

    return np.clip(cce_e + cce_h, 0.0, 1.0)


def collected_charge(
    energy_keV: np.ndarray,
    depth_x: np.ndarray,
    V_bias: float = BIAS_VOLTAGE,
    d: float = DETECTOR_THICKNESS,
    mu_tau_e: float = MU_TAU_E,
    mu_tau_h: float = MU_TAU_H,
) -> np.ndarray:
    """
    Measured charge (in equivalent keV) after charge collection.

    Parameters
    ----------
    energy_keV : np.ndarray
        True photon energies (keV).
    depth_x : np.ndarray
        Interaction depths (cm).  Same shape as energy_keV.

    Returns
    -------
    measured_energy : np.ndarray
        Collected-charge equivalent energy (keV).
    """
    cce = hecht_cce(depth_x, V_bias=V_bias, d=d,
                    mu_tau_e=mu_tau_e, mu_tau_h=mu_tau_h)
    return energy_keV * cce


def attenuation_coefficient(energy_keV: np.ndarray) -> np.ndarray:
    """
    Linear attenuation coefficient of CdTe (cm⁻¹) using a power-law fit
    to NIST XCOM data for 5–50 keV.

    μ(E) ≈ ρ · a · E^(−b),  ρ(CdTe) = 5.85 g/cm³
    """
    rho_CdTe = 5.85      # g/cm³
    a, b = 28.5, 2.80    # fitted coefficients
    E = np.maximum(np.asarray(energy_keV, dtype=float), 0.1)
    mu_rho = a * E ** (-b)   # cm²/g
    return mu_rho * rho_CdTe  # cm⁻¹


def sample_interaction_depth(
    energy_keV: np.ndarray,
    d: float = DETECTOR_THICKNESS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample interaction depths via inverse CDF of Beer-Lambert distribution.

    P(x) = μ · exp(−μx) / (1 − exp(−μd))   for 0 ≤ x ≤ d

    Parameters
    ----------
    energy_keV : np.ndarray
        Photon energies (keV).
    d : float
        Detector thickness (cm).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    depths : np.ndarray  (same shape as energy_keV)
        Sampled interaction depths (cm).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = attenuation_coefficient(energy_keV)
    u = rng.uniform(0.0, 1.0, size=energy_keV.shape)

    # Inverse CDF: x = −ln(1 − u · (1 − exp(−μd))) / μ
    expmud = np.exp(-mu * d)
    depths = -np.log(1.0 - u * (1.0 - expmud)) / np.maximum(mu, 1e-9)
    return np.clip(depths, 0.0, d)
