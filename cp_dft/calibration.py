"""
Electrode potential <-> electron chemical potential.

JDFTx's `target-mu` is an electron chemical potential in HARTREE, referenced
to the calculation's own electrostatic zero. Every reported voltage in this
project passes through here, so the conventions live in one place.

The relation, for mu expressed in eV relative to the same zero the reference
electrode is defined against:

    U_SHE [V]  =  -mu_eV / e  -  PHI_SHE
    mu_eV      =  -(U_SHE + PHI_SHE) * e

with PHI_SHE the absolute potential of the standard hydrogen electrode,
conventionally 4.44 V (Trasatti). Values from 4.28 to 4.6 V appear in the
literature; the choice shifts every potential in the campaign by a constant,
so it must be reported alongside the results.

>>> THIS IS THE PROJECT'S LARGEST SILENT-ERROR RISK. <<<

Two things make the naive formula wrong in practice:

  1. The electrostatic zero of a JDFTx calculation with an implicit fluid is
     the bulk-fluid potential, not the vacuum level. It is close to, but not
     identical with, the vacuum reference the 4.44 V convention assumes.
  2. Sign conventions for `target-mu` differ between codes and between JDFTx
     versions.

So `calibrate_from_reference_slab()` exists, and the honest workflow is:

    a. Run a clean Cu slab whose work function you trust (experimental
       Cu(100) ~ 4.59 eV, Cu(111) ~ 4.94 eV, Cu(110) ~ 4.48 eV).
    b. Read the mu JDFTx reports at the point of zero charge.
    c. Fit the offset with calibrate_from_reference_slab().
    d. Use the returned PotentialCalibration for every subsequent conversion.

An UNCALIBRATED PotentialCalibration is available and will happily convert
numbers, but it is flagged `calibrated=False` and every record derived from
it should be treated as relative, not absolute. Do not report absolute
potentials from an uncalibrated conversion.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

HARTREE_EV = 27.211386245988

# Absolute SHE potential (Trasatti recommendation). Report whichever you use.
PHI_SHE_V_TRASATTI = 4.44
PHI_SHE_V_ALTERNATIVES = {"trasatti": 4.44, "kelvin_probe": 4.28, "upper": 4.60}

# Experimental polycrystalline/facet work functions, eV. For calibration
# targets only -- a DFT work function will differ by ~0.1-0.3 eV depending on
# functional and pseudopotential, which is exactly what the fit absorbs.
CU_WORK_FUNCTION_EV = {"100": 4.59, "110": 4.48, "111": 4.94, "310": 4.45}


@dataclass
class PotentialCalibration:
    """Affine map between JDFTx mu and electrode potential.

        U_SHE = -(mu_ev + offset_ev) / 1.0 - phi_she_v

    `offset_ev` shifts JDFTx's electrostatic zero onto the reference the SHE
    convention assumes. Zero for an uncalibrated instance.
    """

    offset_ev: float = 0.0
    phi_she_v: float = PHI_SHE_V_TRASATTI
    calibrated: bool = False
    reference_facet: Optional[str] = None
    reference_work_function_ev: Optional[float] = None
    reference_mu_ev: Optional[float] = None
    note: str = ""

    # -- conversions -------------------------------------------------------

    def mu_ev_to_u_she(self, mu_ev: float) -> float:
        return -(mu_ev + self.offset_ev) - self.phi_she_v

    def u_she_to_mu_ev(self, u_she_v: float) -> float:
        return -(u_she_v + self.phi_she_v) - self.offset_ev

    def u_she_to_target_mu_hartree(self, u_she_v: float) -> float:
        """What to put in JDFTx's `target-mu` tag for a wanted potential."""
        return self.u_she_to_mu_ev(u_she_v) / HARTREE_EV

    def target_mu_hartree_to_u_she(self, target_mu_hartree: float) -> float:
        return self.mu_ev_to_u_she(target_mu_hartree * HARTREE_EV)

    def mu_ev_to_target_mu_hartree(self, mu_ev: float) -> float:
        return mu_ev / HARTREE_EV

    # -- housekeeping ------------------------------------------------------

    def require_calibrated(self) -> None:
        if not self.calibrated:
            raise RuntimeError(
                "This PotentialCalibration has not been calibrated against a "
                "reference slab, so absolute potentials from it are not "
                "meaningful. Run a clean Cu slab at the point of zero charge "
                "and call calibrate_from_reference_slab(), or explicitly opt "
                "into relative potentials."
            )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PotentialCalibration":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def describe(self) -> str:
        state = "CALIBRATED" if self.calibrated else "UNCALIBRATED (relative only)"
        line = (f"PotentialCalibration [{state}]  offset={self.offset_ev:+.4f} eV  "
                f"phi_SHE={self.phi_she_v:.2f} V")
        if self.calibrated:
            line += (f"\n  reference: Cu({self.reference_facet}), "
                     f"WF={self.reference_work_function_ev:.3f} eV, "
                     f"mu_pzc={self.reference_mu_ev:+.4f} eV")
        if self.note:
            line += f"\n  note: {self.note}"
        return line


UNCALIBRATED = PotentialCalibration(
    note="Default placeholder. Absolute potentials from this are NOT reportable."
)


def calibrate_from_reference_slab(mu_pzc_ev: float, facet: str,
                                  work_function_ev: Optional[float] = None,
                                  phi_she_v: float = PHI_SHE_V_TRASATTI
                                  ) -> PotentialCalibration:
    """Fit the offset from one neutral reference slab.

    At the point of zero charge the electrode potential equals the work
    function referenced to SHE:

        U_pzc = WF - phi_SHE

    Requiring the calibration to reproduce that fixes the offset:

        offset = -mu_pzc - WF
    """
    wf = work_function_ev if work_function_ev is not None else CU_WORK_FUNCTION_EV.get(facet)
    if wf is None:
        raise ValueError(
            f"No reference work function for Cu({facet}). Pass work_function_ev "
            f"explicitly. Known facets: {sorted(CU_WORK_FUNCTION_EV)}."
        )
    offset = -mu_pzc_ev - wf
    return PotentialCalibration(
        offset_ev=offset,
        phi_she_v=phi_she_v,
        calibrated=True,
        reference_facet=facet,
        reference_work_function_ev=wf,
        reference_mu_ev=mu_pzc_ev,
        note=(f"Single-point calibration: Cu({facet}) PZC, "
              f"U_pzc = {wf - phi_she_v:+.3f} V vs SHE."),
    )


def calibrate_from_series(mu_ev_values: Sequence[float],
                          u_she_values: Sequence[float],
                          phi_she_v: float = PHI_SHE_V_TRASATTI
                          ) -> Tuple[PotentialCalibration, Dict[str, float]]:
    """Least-squares calibration over several reference points.

    Also returns the fitted slope. Theory says the slope of U vs mu is exactly
    -1; a fitted slope departing from -1 by more than a few percent means
    something is wrong -- charge leaking into the fluid, an unconverged
    grand-canonical solve, or a sign convention that is not what is assumed
    here. Check the slope before trusting the offset.
    """
    import numpy as np

    mu = np.asarray(mu_ev_values, dtype=float)
    u = np.asarray(u_she_values, dtype=float)
    if mu.size != u.size or mu.size < 2:
        raise ValueError("Need at least two matched (mu, U) pairs of equal length.")

    slope, intercept = np.polyfit(mu, u, 1)
    predicted = slope * mu + intercept
    residual_rms = float(np.sqrt(np.mean((u - predicted) ** 2)))

    # Impose the theoretical slope of -1 and fit only the shift.
    offset = float(-np.mean(u + phi_she_v + mu))
    diagnostics = {
        "fitted_slope": float(slope),
        "slope_deviation_from_minus_one": float(abs(slope + 1.0)),
        "fitted_intercept": float(intercept),
        "residual_rms_v": residual_rms,
        "n_points": int(mu.size),
    }
    note = (f"Least-squares over {mu.size} points; fitted slope "
            f"{slope:+.4f} (theory -1).")
    if abs(slope + 1.0) > 0.05:
        note += "  WARNING: slope departs from -1 by >5% -- investigate before use."
    return PotentialCalibration(
        offset_ev=offset, phi_she_v=phi_she_v, calibrated=True,
        reference_mu_ev=float(np.mean(mu)), note=note,
    ), diagnostics


def potential_grid(u_min_v: float, u_max_v: float, n: int,
                   calibration: PotentialCalibration) -> List[Tuple[float, float]]:
    """Evenly spaced potentials as (U_SHE volts, target-mu hartree) pairs.

    For CO2 reduction on Cu the interesting window is roughly -1.2 to 0.0 V
    vs SHE; the anchor paper's most negative charge density (-23 uC/cm^2)
    sits within it.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n == 1:
        mid = 0.5 * (u_min_v + u_max_v)
        return [(mid, calibration.u_she_to_target_mu_hartree(mid))]
    step = (u_max_v - u_min_v) / (n - 1)
    return [(u_min_v + i * step,
             calibration.u_she_to_target_mu_hartree(u_min_v + i * step))
            for i in range(n)]


def surface_charge_to_potential_estimate(sigma_uc_cm2: float,
                                         capacitance_uf_cm2: float = 20.0,
                                         u_pzc_v: float = -0.2) -> float:
    """Rough U from surface charge via a parallel-plate double layer:

        U = U_pzc + sigma / C

    Only for bridging to the anchor paper's constant-charge state points
    (0 to -30 uC/cm^2) so they can be placed on a potential axis. A constant
    Helmholtz capacitance is a crude model of a real double layer -- treat the
    result as an ordering, not a measurement, and never as a substitute for a
    grand-canonical calculation.
    """
    if capacitance_uf_cm2 <= 0:
        raise ValueError("capacitance_uf_cm2 must be positive.")
    return u_pzc_v + sigma_uc_cm2 / capacitance_uf_cm2
