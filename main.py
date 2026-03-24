"""
Entry point for the CdTe detector spectral distortion simulation.

Run with:
    python main.py

The GUI allows you to control:
    - X-ray tube voltage (20–50 kV)
    - X-ray tube current (10–200 μA)
    - CdTe detector bias voltage (200–1000 V)
    - Number of spectra to generate
    - Acquisition time per spectrum (s)

The simulation models:
    1. Bremsstrahlung + W L-line X-ray tube spectrum (photon source)
    2. Beer-Lambert photon interaction depth sampling
    3. Hecht-equation charge collection efficiency (charge transport)
    4. Space-charge polarisation degradation (polarisation)
    5. Gaussian shaping noise / electronic broadening
    6. Pulse pile-up (full and partial)
    7. Non-paralyzable dead time

Output: true (ideal) spectrum and distorted (measured) spectrum side by side.
"""

import sys
import os

# Ensure the package root is on sys.path when launched as a script
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gui.app import SimulationApp


def main() -> None:
    app = SimulationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
