"""
Dynamic photon event-stream processor.

This module is the top-level simulation engine.  It:

1. Draws a random photon event stream from the source spectrum (Poisson arrivals).
2. Applies Beer-Lambert depth sampling for each photon.
3. Applies the Hecht charge-collection correction (possibly under polarisation).
4. Adds Gaussian electronic noise / broadening.
5. Applies pile-up processing.
6. Applies dead-time rejection.
7. Histograms the events into the requested energy bins to produce both the
   **true** spectrum and the **distorted** (measured) spectrum.

Usage
-----
    from simulation.event_stream import run_simulation

    energy_bins, true_spec, distorted_spec = run_simulation(
        voltage_kv=40,
        current_ua=100,
        n_spectra=1,
        acquisition_time_s=1.0,
    )
"""

from __future__ import annotations

import numpy as np

from .photon_source import generate_spectrum
from .charge_transport import (
    DETECTOR_THICKNESS,
    BIAS_VOLTAGE,
    collected_charge,
    sample_interaction_depth,
)
from .polarization import effective_bias, polarisation_factor
from .electronics import (
    TAU_SHAPING_S,
    TAU_DEAD_S,
    apply_gaussian_broadening,
    simulate_pileup,
    apply_dead_time,
)

# ─── Default energy axis ───────────────────────────────────────────────────────
E_MIN_KEV: float = 1.0
E_MAX_KEV: float = 55.0
N_BINS: int = 512


def _make_energy_axis(
    e_min: float = E_MIN_KEV,
    e_max: float = E_MAX_KEV,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """Return bin centre energies (keV)."""
    return np.linspace(e_min, e_max, n_bins)


def run_simulation(
    voltage_kv: float,
    current_ua: float,
    n_spectra: int = 1,
    acquisition_time_s: float = 1.0,
    bias_voltage: float = BIAS_VOLTAGE,
    detector_thickness: float = DETECTOR_THICKNESS,
    tau_shaping: float = TAU_SHAPING_S,
    tau_dead: float = TAU_DEAD_S,
    integration_time_s: float = 60.0,
    energy_bins: np.ndarray | None = None,
    seed: int | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a full CdTe detector Monte-Carlo simulation.

    Parameters
    ----------
    voltage_kv : float
        X-ray tube voltage (kV, 20–50).
    current_ua : float
        X-ray tube current (μA, 10–200).
    n_spectra : int
        Number of independent spectra to accumulate (result is summed).
    acquisition_time_s : float
        Acquisition time per spectrum (s).
    bias_voltage : float
        CdTe detector bias voltage (V).
    detector_thickness : float
        Detector thickness (cm).
    tau_shaping : float
        Shaper peaking time (s).
    tau_dead : float
        Per-event dead time (s).
    integration_time_s : float
        Total irradiation time for polarisation model (s).
    energy_bins : np.ndarray or None
        Custom bin-centre array (keV).  If None, default axis is used.
    seed : int or None
        Random seed for reproducibility.
    progress_callback : callable(float) or None
        Called with fraction complete [0, 1] after each spectrum.

    Returns
    -------
    energy_bins : np.ndarray
        Bin centre energies (keV).
    true_spectrum : np.ndarray
        Summed true (ideal) spectrum counts per bin.
    distorted_spectrum : np.ndarray
        Summed distorted (measured) spectrum counts per bin.
    """
    rng = np.random.default_rng(seed)

    if energy_bins is None:
        energy_bins = _make_energy_axis()

    bin_width = energy_bins[1] - energy_bins[0]
    bin_edges = np.append(energy_bins - bin_width / 2,
                          energy_bins[-1] + bin_width / 2)

    # ── Source spectrum (photons / keV / s) at detector entrance ────────────
    source_spectrum = generate_spectrum(
        energy_bins, voltage_kv, current_ua
    )
    # Total expected count rate (cps)
    total_rate = float(source_spectrum.sum() * bin_width)
    # Probability of each energy bin
    prob = source_spectrum / (source_spectrum.sum() + 1e-30)

    # ── Polarisation: compute effective bias voltage ─────────────────────────
    flux_cps = total_rate
    pol_frac = polarisation_factor(flux_cps, integration_time_s)
    V_eff = max(bias_voltage * (1.0 - pol_frac), 1.0)

    true_spectrum = np.zeros(len(energy_bins))
    distorted_spectrum = np.zeros(len(energy_bins))

    for spec_idx in range(n_spectra):
        # ── Sample number of photons for this acquisition (Poisson) ─────────
        n_photons = rng.poisson(total_rate * acquisition_time_s)
        if n_photons == 0:
            if progress_callback:
                progress_callback((spec_idx + 1) / n_spectra)
            continue

        # ── Sample photon energies from source distribution ─────────────────
        bin_indices = rng.choice(len(energy_bins), size=n_photons, p=prob)
        photon_energies_true = energy_bins[bin_indices]

        # ── Build true spectrum (ideal, no detector effects) ─────────────────
        counts_true, _ = np.histogram(photon_energies_true, bins=bin_edges)
        true_spectrum += counts_true

        # ── Sample Poisson inter-arrival times → event timeline ─────────────
        # Mean interval = 1/rate; sum of n exponentials gives total time
        inter_arrivals = rng.exponential(1.0 / max(total_rate, 1e-9),
                                         size=n_photons)
        photon_times = np.cumsum(inter_arrivals)

        # ── Sample interaction depth from Beer-Lambert distribution ──────────
        depths = sample_interaction_depth(photon_energies_true,
                                          d=detector_thickness, rng=rng)

        # ── Apply charge transport (Hecht equation) ──────────────────────────
        energies_cc = collected_charge(
            photon_energies_true, depths,
            V_bias=V_eff, d=detector_thickness,
        )

        # ── Apply Gaussian electronic broadening ─────────────────────────────
        energies_broad = apply_gaussian_broadening(energies_cc, rng=rng)

        # ── Apply pile-up ────────────────────────────────────────────────────
        times_pu, energies_pu = simulate_pileup(
            photon_times, energies_broad,
            tau_shaping=tau_shaping, rng=rng,
        )

        # ── Apply dead-time ──────────────────────────────────────────────────
        _, energies_dt = apply_dead_time(times_pu, energies_pu,
                                         tau_dead=tau_dead)

        # ── Histogram measured events ─────────────────────────────────────────
        counts_dist, _ = np.histogram(energies_dt, bins=bin_edges)
        distorted_spectrum += counts_dist

        if progress_callback:
            progress_callback((spec_idx + 1) / n_spectra)

    return energy_bins, true_spectrum, distorted_spectrum


def simulation_metadata(
    voltage_kv: float,
    current_ua: float,
    acquisition_time_s: float = 1.0,
    bias_voltage: float = BIAS_VOLTAGE,
    integration_time_s: float = 60.0,
) -> dict:
    """
    Return a dictionary of simulation parameters and derived quantities for
    display in the GUI.
    """
    energy_bins = _make_energy_axis()
    bin_width = energy_bins[1] - energy_bins[0]
    source_spectrum = generate_spectrum(energy_bins, voltage_kv, current_ua)
    total_rate = float(source_spectrum.sum() * bin_width)
    pol_frac = polarisation_factor(total_rate, integration_time_s)
    V_eff = max(bias_voltage * (1.0 - pol_frac), 1.0)

    return {
        "tube_voltage_kv": voltage_kv,
        "tube_current_ua": current_ua,
        "bias_voltage_V": bias_voltage,
        "effective_bias_V": V_eff,
        "polarisation_fraction": pol_frac,
        "photon_rate_cps": total_rate,
        "acquisition_time_s": acquisition_time_s,
        "expected_counts": total_rate * acquisition_time_s,
    }
