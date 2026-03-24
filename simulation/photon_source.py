"""
X-ray tube spectrum model.

Generates the Bremsstrahlung continuum (modified Kramers formula) and
the tungsten anode characteristic L-lines for tube voltages in the 20–50 kV
range.  Tube current (μA) scales the total photon flux linearly.

Physical model
--------------
Kramers formula:
    N(E) ∝ I · Z · (E_max/E − 1)    for 0 < E < E_max
    N(E) = 0                         for E ≥ E_max

with spectral filtration by:
    - 1 mm aluminium (half-value layer) inherent filtration
    - 0.5 mm beryllium window

Tungsten L-lines relevant below 50 kV (energies in keV):
    Lα₁  8.398 keV,  Lβ₁  9.672 keV,  Lγ₁  11.286 keV
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ─── Tungsten atomic number ───────────────────────────────────────────────────
_Z_W = 74

# ─── Tungsten L-line energies (keV) and relative intensities ─────────────────
_W_L_LINES: list[tuple[float, float]] = [
    (8.398, 1.00),   # Lα₁
    (9.672, 0.55),   # Lβ₁
    (11.286, 0.12),  # Lγ₁
]

# ─── Mass attenuation coefficients (cm²/g) fitted as μ/ρ = a·E^(-b) ──────────
# Aluminium: valid ~5–50 keV
_AL_DENSITY = 2.70   # g/cm³
_AL_THICKNESS = 0.1  # cm (1 mm)
_BE_DENSITY = 1.848  # g/cm³
_BE_THICKNESS = 0.05 # cm (0.5 mm)


def _mass_attn_al(E_keV: np.ndarray) -> np.ndarray:
    """Approximate Al mass-attenuation coefficient (cm²/g) via power law."""
    # Fitted to NIST XCOM data, 5–50 keV
    a, b = 5.31, 2.72
    return a * E_keV ** (-b)


def _mass_attn_be(E_keV: np.ndarray) -> np.ndarray:
    """Approximate Be mass-attenuation coefficient (cm²/g) via power law."""
    a, b = 0.68, 2.60
    return a * E_keV ** (-b)


def _filtration_transmission(E_keV: np.ndarray) -> np.ndarray:
    """Combined Al + Be transmission factor (Beer-Lambert)."""
    mu_al = _mass_attn_al(E_keV) * _AL_DENSITY * _AL_THICKNESS
    mu_be = _mass_attn_be(E_keV) * _BE_DENSITY * _BE_THICKNESS
    return np.exp(-mu_al - mu_be)


def generate_spectrum(
    energy_bins: np.ndarray,
    voltage_kv: float,
    current_ua: float,
    detector_efficiency: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate the true X-ray photon-count spectrum at the detector entrance.

    Parameters
    ----------
    energy_bins : np.ndarray
        Centre energies of each bin, in **keV** (1-D, monotone increasing).
    voltage_kv : float
        Tube peak voltage in kV (20–50).
    current_ua : float
        Tube current in μA (10–200).
    detector_efficiency : np.ndarray or None
        Optional per-bin detection efficiency [0–1].  If None, all ones.

    Returns
    -------
    spectrum : np.ndarray
        Photon count spectrum (arbitrary but consistent units).
    """
    E = np.asarray(energy_bins, dtype=float)
    E_max = float(voltage_kv)  # keV

    # ── Bremsstrahlung continuum (Kramers) ─────────────────────────────────
    spectrum = np.where(
        E < E_max,
        _Z_W * (E_max / np.maximum(E, 1e-9) - 1.0),
        0.0,
    )

    # ── Add tungsten L-characteristic lines ────────────────────────────────
    # σ = 200 eV (Gaussian natural line width + pre-detector broadening)
    sigma_keV = 0.20
    for E_line, rel_int in _W_L_LINES:
        if E_line >= E_max:
            continue
        # Build a Gaussian peak centred on the line energy
        line = rel_int * np.exp(-0.5 * ((E - E_line) / sigma_keV) ** 2)
        # Scale so that the line area equals (fraction of Kramers at E_max/2)
        scale = 0.15 * _Z_W * (E_max / max(E_line, 1e-9) - 1.0)
        spectrum += scale * line / (line.sum() + 1e-30) * len(E)

    # ── Apply spectral filtration ──────────────────────────────────────────
    transmission = _filtration_transmission(np.maximum(E, 0.1))
    spectrum *= transmission

    # ── Scale by tube current (flux ∝ I) ──────────────────────────────────
    flux_scale = current_ua / 100.0  # normalise to 100 μA baseline
    spectrum *= flux_scale

    # ── Optional detector efficiency ──────────────────────────────────────
    if detector_efficiency is not None:
        spectrum *= np.asarray(detector_efficiency)

    return spectrum
