"""
Polarization model for CdTe detectors.

Under continuous irradiation, trapped holes form a positive space-charge
layer near the cathode, which reduces (and eventually reverses) the internal
electric field.  This causes:
    - Reduction of effective bias voltage felt by charge carriers
    - Increased low-energy tailing in the measured spectrum

Simplified phenomenological model
----------------------------------
The effective bias is modelled as:

    V_eff(Φ, t) = V_bias · [1 − p_max · (1 − exp(−t/τ_pol))]

where:
    Φ       = incident photon flux  (counts/s)
    p_max   = maximum fractional polarisation  (0 → 1)
    τ_pol   = polarisation time constant (s) — depends on flux
    t       = integration time (s)

The maximum polarisation fraction and time constant scale with flux:
    p_max = min(Φ / Φ_sat, 1)     Φ_sat = saturation flux
    τ_pol = τ_0 / (1 + Φ / Φ_0)  τ_0 = 300 s,  Φ_0 = 1e5 cps

References: Sellin et al. (2005), Bale & Szeles (2008).
"""

from __future__ import annotations

import numpy as np

# ─── Default polarisation model parameters ───────────────────────────────────
PHI_SAT: float = 5.0e5     # saturation flux (cps) beyond which p_max → 1
TAU_0: float = 300.0       # base polarisation time constant (s)
PHI_0: float = 1.0e5       # reference flux (cps) for τ scaling
MAX_POL_FRACTION: float = 0.60  # physical upper bound (V_eff ≥ 0.4 · V_bias)


def polarisation_factor(
    flux_cps: float,
    integration_time_s: float = 60.0,
    phi_sat: float = PHI_SAT,
    tau_0: float = TAU_0,
    phi_0: float = PHI_0,
    max_pol: float = MAX_POL_FRACTION,
) -> float:
    """
    Return the fractional reduction in effective bias due to polarisation.

    Parameters
    ----------
    flux_cps : float
        Total incident photon flux in counts/second.
    integration_time_s : float
        Elapsed irradiation time in seconds.
    phi_sat, tau_0, phi_0, max_pol : float
        Model parameters (see module docstring).

    Returns
    -------
    pol_frac : float
        Polarisation fraction in [0, max_pol].
        V_eff = V_bias · (1 − pol_frac).
    """
    p_max = min(flux_cps / phi_sat, 1.0) * max_pol
    tau_pol = tau_0 / (1.0 + flux_cps / phi_0)
    pol_frac = p_max * (1.0 - np.exp(-integration_time_s / tau_pol))
    return float(np.clip(pol_frac, 0.0, max_pol))


def effective_bias(
    V_bias: float,
    flux_cps: float,
    integration_time_s: float = 60.0,
    **kwargs,
) -> float:
    """
    Effective detector bias voltage after polarisation.

    Parameters
    ----------
    V_bias : float
        Nominal detector bias (V).
    flux_cps : float
        Incident photon flux (cps).
    integration_time_s : float
        Irradiation time (s).

    Returns
    -------
    V_eff : float
        Effective bias (V), always > 0.
    """
    pf = polarisation_factor(flux_cps, integration_time_s, **kwargs)
    return max(V_bias * (1.0 - pf), 1.0)


def polarisation_noise_sigma(
    energy_keV: float,
    pol_frac: float,
    sigma_base_keV: float = 0.15,
) -> float:
    """
    Additional energy smearing caused by inhomogeneous field from polarisation.

    The smearing is proportional to polarisation fraction and photon energy
    (higher-energy photons traverse more of the non-uniform field region).

    Parameters
    ----------
    energy_keV : float
        Photon energy (keV).
    pol_frac : float
        Polarisation fraction (0–1).
    sigma_base_keV : float
        Baseline electronic-noise sigma (keV).

    Returns
    -------
    sigma : float
        Combined energy resolution sigma (keV).
    """
    sigma_pol = 0.03 * pol_frac * energy_keV  # 3 % per unit polarisation
    return float(np.sqrt(sigma_base_keV**2 + sigma_pol**2))
