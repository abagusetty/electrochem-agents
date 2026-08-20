"""
Acquisition policies: which state points and which frames earn an expensive
grand-canonical DFT label.

This package is the project's headline scientific contribution. Everything
else exists so that the comparison it defines is meaningful.

  acquisition.policies  -- the baselines the headline claim is measured
                           against: B0 grid, B1 random, B2 force-only.
  acquisition.sigma_mu  -- A1, the proposed policy: trigger on committee
                           disagreement in the FERMI LEVEL.
  acquisition.registry  -- pre-registration of the comparison, so the
                           baseline cannot be chosen after seeing results.

The falsification test is built in. If A1 does not beat B2, sigma_mu carries
no information beyond ordinary force uncertainty, and that is the finding.
"""

from acquisition.policies import (
    AcquisitionPolicy, GridPolicy, RandomPolicy, ForceUncertaintyPolicy,
    Candidate, AcquisitionDecision,
)
from acquisition.sigma_mu import SigmaMuPolicy, SigmaMuConfig, ThresholdCalibration
from acquisition.registry import PreRegistration, compare_policies

__all__ = [
    "AcquisitionPolicy", "GridPolicy", "RandomPolicy", "ForceUncertaintyPolicy",
    "Candidate", "AcquisitionDecision",
    "SigmaMuPolicy", "SigmaMuConfig", "ThresholdCalibration",
    "PreRegistration", "compare_policies",
]
