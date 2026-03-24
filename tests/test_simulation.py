"""
Unit tests for the CdTe detector simulation framework.

Tests cover:
    - photon_source: spectral shape and flux scaling
    - charge_transport: Hecht CCE bounds, depth sampling
    - polarization: polarisation fraction bounds
    - electronics: dead-time reduction, pileup energy conservation
    - event_stream: full simulation sanity checks
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.photon_source import generate_spectrum
from simulation.charge_transport import (
    hecht_cce, sample_interaction_depth, collected_charge,
    DETECTOR_THICKNESS, BIAS_VOLTAGE,
)
from simulation.polarization import polarisation_factor, effective_bias
from simulation.electronics import (
    energy_resolution_sigma, apply_gaussian_broadening,
    simulate_pileup, apply_dead_time,
    TAU_SHAPING_S, TAU_DEAD_S,
)
from simulation.event_stream import run_simulation, simulation_metadata


# ─── Helpers ──────────────────────────────────────────────────────────────────

ENERGY_BINS = np.linspace(1.0, 55.0, 256)


def spectrum_excluding(spec, center, window):
    """Return spec values excluding [center-window, center+window]."""
    mask = np.ones(len(spec), dtype=bool)
    mask[max(0, center - window): center + window + 1] = False
    return spec[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# photon_source tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhotonSource:
    def test_spectrum_non_negative(self):
        spec = generate_spectrum(ENERGY_BINS, voltage_kv=40, current_ua=100)
        assert np.all(spec >= 0), "Spectrum must be non-negative"

    def test_spectrum_zero_above_voltage(self):
        kv = 30.0
        spec = generate_spectrum(ENERGY_BINS, voltage_kv=kv, current_ua=100)
        above = spec[ENERGY_BINS > kv]
        assert np.allclose(above, 0.0), (
            "Spectrum must be zero above tube voltage"
        )

    def test_flux_scales_with_current(self):
        spec_lo = generate_spectrum(ENERGY_BINS, 40, 50)
        spec_hi = generate_spectrum(ENERGY_BINS, 40, 200)
        ratio = spec_hi.sum() / spec_lo.sum()
        # Current increased by 4× (50 → 200 μA); flux ratio should be ≈4
        assert 3.8 < ratio < 4.2, f"Expected ~4× flux ratio, got {ratio:.3f}"

    def test_higher_voltage_extends_spectrum(self):
        spec_lo = generate_spectrum(ENERGY_BINS, 25, 100)
        spec_hi = generate_spectrum(ENERGY_BINS, 50, 100)
        # Higher kV → more total counts
        assert spec_hi.sum() > spec_lo.sum()

    def test_l_lines_present(self):
        """W L-alpha line (~8.4 keV) should be visible at 40 kV."""
        spec = generate_spectrum(ENERGY_BINS, 40, 100)
        idx = np.argmin(np.abs(ENERGY_BINS - 8.4))
        # Value at 8.4 keV should be higher than the mean away from it
        assert spec[idx] > np.mean(spectrum_excluding(spec, idx, window=10))

    def test_detector_efficiency_applied(self):
        eff = np.ones(len(ENERGY_BINS)) * 0.5
        spec_no_eff = generate_spectrum(ENERGY_BINS, 40, 100)
        spec_with_eff = generate_spectrum(ENERGY_BINS, 40, 100,
                                          detector_efficiency=eff)
        np.testing.assert_allclose(spec_with_eff, spec_no_eff * 0.5,
                                   rtol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# charge_transport tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChargeTransport:
    def test_cce_bounds(self):
        depths = np.linspace(0, DETECTOR_THICKNESS, 100)
        cce = hecht_cce(depths)
        assert np.all(cce >= 0) and np.all(cce <= 1.0)

    def test_cce_decreases_with_reduced_bias(self):
        depths = np.linspace(0, DETECTOR_THICKNESS, 50)
        cce_high = hecht_cce(depths, V_bias=800)
        cce_low = hecht_cce(depths, V_bias=200)
        assert cce_high.mean() > cce_low.mean()

    def test_depth_sampling_in_bounds(self):
        rng = np.random.default_rng(0)
        E = np.full(500, 30.0)
        depths = sample_interaction_depth(E, rng=rng)
        assert np.all(depths >= 0.0)
        assert np.all(depths <= DETECTOR_THICKNESS)

    def test_collected_charge_le_true_energy(self):
        rng = np.random.default_rng(1)
        E = np.full(200, 25.0)
        depths = sample_interaction_depth(E, rng=rng)
        E_coll = collected_charge(E, depths)
        assert np.all(E_coll <= E + 1e-12)

    def test_collected_charge_positive(self):
        rng = np.random.default_rng(2)
        E = np.random.default_rng(2).uniform(5, 50, 300)
        depths = sample_interaction_depth(E, rng=rng)
        E_coll = collected_charge(E, depths)
        assert np.all(E_coll >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# polarization tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolarization:
    def test_pol_fraction_in_bounds(self):
        for flux in [1e3, 1e5, 1e6]:
            pf = polarisation_factor(flux, 60.0)
            assert 0.0 <= pf <= 0.60, (
                f"pol_fraction={pf} out of [0, 0.60] at flux={flux}"
            )

    def test_pol_increases_with_flux(self):
        pf_lo = polarisation_factor(1e3, 300)
        pf_hi = polarisation_factor(1e6, 300)
        assert pf_hi > pf_lo

    def test_effective_bias_lt_nominal(self):
        V_eff = effective_bias(600.0, 1e6, 300)
        assert V_eff < 600.0

    def test_effective_bias_always_positive(self):
        V_eff = effective_bias(600.0, 1e9, 3600)
        assert V_eff > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# electronics tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestElectronics:
    def test_energy_resolution_positive(self):
        E = np.linspace(5, 50, 20)
        sigma = energy_resolution_sigma(E)
        assert np.all(sigma > 0)

    def test_energy_resolution_increases_with_energy(self):
        sig_lo = energy_resolution_sigma(10.0)
        sig_hi = energy_resolution_sigma(50.0)
        assert sig_hi > sig_lo

    def test_gaussian_broadening_preserves_count(self):
        rng = np.random.default_rng(3)
        E = rng.uniform(5, 50, 1000)
        E_broad = apply_gaussian_broadening(E, rng=rng)
        assert len(E_broad) == len(E)

    def test_pileup_reduces_event_count(self):
        rng = np.random.default_rng(4)
        n = 500
        times = np.sort(rng.uniform(0, 100e-6, n))
        energies = rng.uniform(10, 40, n)
        t_out, e_out = simulate_pileup(times, energies,
                                       tau_shaping=TAU_SHAPING_S, rng=rng)
        assert len(t_out) <= n

    def test_pileup_full_sums_energy(self):
        """Full pileup of two equal-energy events → one event with 2× energy."""
        dt = TAU_SHAPING_S * 0.1
        times = np.array([0.0, dt])
        energies = np.array([20.0, 20.0])
        rng = np.random.default_rng(5)
        t_out, e_out = simulate_pileup(times, energies,
                                       tau_shaping=TAU_SHAPING_S, rng=rng)
        assert len(e_out) == 1
        assert abs(e_out[0] - 40.0) < 1e-9

    def test_dead_time_never_increases_count(self):
        rng = np.random.default_rng(6)
        n = 1000
        times = np.sort(rng.uniform(0, 1e-3, n))
        energies = rng.uniform(10, 40, n)
        t_out, e_out = apply_dead_time(times, energies, tau_dead=TAU_DEAD_S)
        assert len(t_out) <= n

    def test_dead_time_minimum_spacing(self):
        rng = np.random.default_rng(7)
        n = 500
        times = np.sort(rng.uniform(0, 5e-4, n))
        energies = rng.uniform(5, 40, n)
        t_out, _ = apply_dead_time(times, energies, tau_dead=TAU_DEAD_S)
        if len(t_out) > 1:
            diffs = np.diff(t_out)
            assert np.all(diffs >= TAU_DEAD_S - 1e-15), (
                "Dead-time spacing violated"
            )

    def test_empty_event_stream(self):
        times = np.array([])
        energies = np.array([])
        t_pu, e_pu = simulate_pileup(times, energies)
        t_dt, e_dt = apply_dead_time(times, energies)
        assert len(t_pu) == 0
        assert len(t_dt) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# event_stream / integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventStream:
    def test_output_shapes(self):
        e_bins, true_s, dist_s = run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=1,
            acquisition_time_s=0.05, seed=0,
        )
        assert e_bins.shape == true_s.shape == dist_s.shape
        assert len(e_bins) > 0

    def test_spectra_non_negative(self):
        _, true_s, dist_s = run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=1,
            acquisition_time_s=0.05, seed=1,
        )
        assert np.all(true_s >= 0)
        assert np.all(dist_s >= 0)

    def test_distorted_count_reasonable(self):
        """Distorted count should not dramatically exceed true count."""
        _, true_s, dist_s = run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=3,
            acquisition_time_s=0.2, seed=2,
        )
        t = true_s.sum()
        d = dist_s.sum()
        if t > 0:
            assert d / t < 2.0, f"Distorted/True ratio {d/t:.2f} unexpectedly high"

    def test_higher_current_more_counts(self):
        _, true_lo, _ = run_simulation(
            voltage_kv=40, current_ua=10, n_spectra=1,
            acquisition_time_s=1.0, seed=3,
        )
        _, true_hi, _ = run_simulation(
            voltage_kv=40, current_ua=200, n_spectra=1,
            acquisition_time_s=1.0, seed=3,
        )
        assert true_hi.sum() > true_lo.sum()

    def test_n_spectra_accumulates(self):
        _, spec1, _ = run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=1,
            acquisition_time_s=1.0, seed=10,
        )
        _, spec5, _ = run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=5,
            acquisition_time_s=1.0, seed=20,
        )
        if spec1.sum() > 0:
            ratio = spec5.sum() / spec1.sum()
            assert 1.0 < ratio < 15.0, f"n_spectra ratio {ratio:.2f} unexpected"

    def test_progress_callback(self):
        fractions = []

        def cb(f):
            fractions.append(f)

        run_simulation(
            voltage_kv=40, current_ua=100, n_spectra=3,
            acquisition_time_s=0.05, seed=11,
            progress_callback=cb,
        )
        assert len(fractions) == 3
        assert fractions[-1] == pytest.approx(1.0)

    def test_metadata_keys(self):
        meta = simulation_metadata(40, 100, 1.0)
        required_keys = {
            "tube_voltage_kv", "tube_current_ua", "bias_voltage_V",
            "effective_bias_V", "polarisation_fraction",
            "photon_rate_cps", "expected_counts",
        }
        assert required_keys.issubset(set(meta.keys()))

    def test_metadata_physical_consistency(self):
        meta = simulation_metadata(40, 100, 1.0)
        assert meta["effective_bias_V"] <= meta["bias_voltage_V"]
        assert meta["polarisation_fraction"] >= 0.0
        assert meta["photon_rate_cps"] > 0
