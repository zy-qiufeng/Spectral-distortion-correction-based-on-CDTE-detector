"""
Electronics effects for CdTe detector readout.

Models the following signal-processing artefacts:

1. **Gaussian shaping** – bandpass filter with peaking time τ_shaping.
   The energy resolution is modelled as a Gaussian broadening in energy space:
       FWHM(E) = √(FWHM_noise² + FWHM_Fano²)
       FWHM_Fano = 2.355 · √(F · E / ε)
       FWHM_noise ≈ ENC contribution

2. **Pileup** – when two photons arrive within the shaper peaking time,
   their pulses partially or fully overlap.  Three regimes are modelled:
       - No pileup     (Δt > τ_shaping)  → normal processing
       - Partial pileup (τ_shaping/2 < Δt ≤ τ_shaping) → sum pulse counted twice
       - Full pileup   (Δt ≤ τ_shaping/2) → sum pulse counted once

3. **Dead time** – non-paralyzable (Type I, Geiger-Müller) dead time:
       n_obs = n_true / (1 + n_true · τ_dead)
   Applied by randomly discarding events that fall within τ_dead of the
   previous accepted event.

Physical parameters
-------------------
    τ_shaping = 500 ns  (shaper peaking time)
    τ_dead    = 1.0 μs  (total per-event dead time, including reset)
    F         = 0.14    (Fano factor for CdTe)
    ε (CdTe)  = 4.43 eV/pair
    ENC       = 50 e⁻  (equivalent noise charge, RMS)
"""

from __future__ import annotations

import numpy as np

# ─── Default electronic parameters ───────────────────────────────────────────
TAU_SHAPING_S: float = 500e-9   # s – shaper peaking time
TAU_DEAD_S: float = 1.0e-6      # s – dead time
FANO_FACTOR: float = 0.14
E_EH_EV: float = 4.43           # eV per electron-hole pair
ENC_ELECTRONS: float = 50.0     # RMS equivalent noise charge (electrons)


def energy_resolution_sigma(
    energy_keV: float | np.ndarray,
    enc: float = ENC_ELECTRONS,
    fano: float = FANO_FACTOR,
    e_eh_eV: float = E_EH_EV,
) -> np.ndarray:
    """
    Gaussian sigma of energy resolution (keV) at given photon energy.

    σ = FWHM / 2.355,  FWHM² = FWHM_noise² + FWHM_Fano²

    Parameters
    ----------
    energy_keV : float or np.ndarray
        Photon energy in keV.
    enc : float
        Equivalent noise charge (e⁻ RMS).
    fano : float
        Fano factor for CdTe.
    e_eh_eV : float
        Energy per e-h pair (eV).

    Returns
    -------
    sigma_keV : np.ndarray
    """
    E_eV = np.asarray(energy_keV, dtype=float) * 1e3
    # Noise contribution
    fwhm_noise_eV = 2.355 * enc * e_eh_eV
    # Fano (statistical) contribution
    fwhm_fano_eV = 2.355 * np.sqrt(fano * E_eV * e_eh_eV)
    fwhm_total_eV = np.sqrt(fwhm_noise_eV**2 + fwhm_fano_eV**2)
    return fwhm_total_eV / 2.355 / 1e3  # → keV σ


def apply_gaussian_broadening(
    energies: np.ndarray,
    enc: float = ENC_ELECTRONS,
    fano: float = FANO_FACTOR,
    e_eh_eV: float = E_EH_EV,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply per-event Gaussian energy broadening to a list of photon energies.

    Parameters
    ----------
    energies : np.ndarray
        True (or charge-corrected) energies in keV.
    rng : np.random.Generator or None

    Returns
    -------
    broadened : np.ndarray
        Measured energies after electronic broadening (keV).
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = energy_resolution_sigma(energies, enc=enc, fano=fano, e_eh_eV=e_eh_eV)
    noise = rng.normal(0.0, sigma)
    return energies + noise


def simulate_pileup(
    photon_times: np.ndarray,
    photon_energies: np.ndarray,
    tau_shaping: float = TAU_SHAPING_S,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply pile-up to an ordered stream of photon events.

    Photon events are processed sequentially.  When the inter-arrival time
    Δt between consecutive events is ≤ τ_shaping:
        - Δt ≤ τ_shaping/2 : full pileup → one event recorded with summed energy
        - τ_shaping/2 < Δt ≤ τ_shaping : partial pileup → two events each
          recorded with slightly raised baseline (adds 30 % of the companion
          energy to each), approximating the analogue sum in the shaper output.

    Parameters
    ----------
    photon_times : np.ndarray
        Event arrival times (s), sorted ascending.
    photon_energies : np.ndarray
        Photon energies (keV), same length as photon_times.
    tau_shaping : float
        Shaper peaking time (s).

    Returns
    -------
    out_times, out_energies : np.ndarray
        Processed event times and energies.
    """
    if len(photon_times) == 0:
        return photon_times.copy(), photon_energies.copy()

    if rng is None:
        rng = np.random.default_rng()

    times = np.asarray(photon_times, dtype=float)
    energies = np.asarray(photon_energies, dtype=float)

    out_times: list[float] = []
    out_energies: list[float] = []

    i = 0
    n = len(times)
    while i < n:
        if i + 1 < n:
            dt = times[i + 1] - times[i]
            if dt <= tau_shaping / 2.0:
                # Full pileup: record one summed event at the earlier time
                out_times.append(times[i])
                out_energies.append(energies[i] + energies[i + 1])
                i += 2
                continue
            elif dt <= tau_shaping:
                # Partial pileup: pulses overlap by fraction (1 - dt/τ).
                # Each event's measured energy is raised by frac × companion energy,
                # approximating the analogue sum on the shaper output.
                frac = 1.0 - dt / tau_shaping   # fractional contribution from companion pulse
                out_times.append(times[i])
                out_energies.append(energies[i] + frac * energies[i + 1])
                out_times.append(times[i + 1])
                out_energies.append(energies[i + 1] + frac * energies[i])
                i += 2
                continue
        out_times.append(times[i])
        out_energies.append(energies[i])
        i += 1

    return np.array(out_times), np.array(out_energies)


def apply_dead_time(
    photon_times: np.ndarray,
    photon_energies: np.ndarray,
    tau_dead: float = TAU_DEAD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply non-paralyzable (Type I) dead time to an ordered event stream.

    Events that arrive within τ_dead of the last accepted event are rejected.

    Parameters
    ----------
    photon_times : np.ndarray
        Sorted event arrival times (s).
    photon_energies : np.ndarray
        Energies (keV).
    tau_dead : float
        Dead-time window (s).

    Returns
    -------
    accepted_times, accepted_energies : np.ndarray
    """
    if len(photon_times) == 0:
        return photon_times.copy(), photon_energies.copy()

    times = np.asarray(photon_times, dtype=float)
    energies = np.asarray(photon_energies, dtype=float)

    accepted_mask = np.zeros(len(times), dtype=bool)
    last_accept = -np.inf
    for idx in range(len(times)):
        if times[idx] - last_accept >= tau_dead:
            accepted_mask[idx] = True
            last_accept = times[idx]

    return times[accepted_mask], energies[accepted_mask]
