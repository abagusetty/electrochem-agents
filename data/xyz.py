"""
Export CP-DFT labels to CP-MACE's extended-XYZ training format.

Format confirmed from github.com/yuanyue-liu-group/CP-MACE (README and
simulation/slow_growth/init.xyz):

    <n_atoms>
    Lattice="ax ay az bx by bz cx cy cz" Properties=species:S:1:pos:R:3:forces:R:3 \
      energy=<E> electron=<N_e> potential=<mu> pbc="T T T"
    Cu  x y z  fx fy fz
    ...

The two tags that make it constant-potential:
    electron=<N_e>   number of electrons in the cell
    potential=<mu>   Fermi level

Upstream example values: `electron=661.7`, `potential=-3.407347`.

>>> UNIT AND SIGN CONVENTION -- VERIFY BEFORE TRAINING <<<
CP-MACE's `potential` is a Fermi level, and JDFTx reports `mu` in Hartree
internally while pymatgen exposes eV. This writer assumes BOTH are eV and
does not convert. Before training on a real dataset, check that the exported
`potential=` values sit in the same numeric range as upstream's example
(order -3 to -5 eV for a metal). A factor-of-27.2 error here would train a
model that fits perfectly and predicts a meaningless potential.

Sign convention for `target-mu` vs. absolute electrode potential is a
separate question, handled in cp_dft.calibration.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from data.schema import CPDFTLabel


def _format_frame(label: CPDFTLabel,
                  include_forces: bool = True,
                  extra_info: Optional[dict] = None) -> str:
    n_atoms = label.n_atoms
    if n_atoms == 0:
        raise ValueError(
            f"Label {label.state_point_id} has no atoms; it cannot be written "
            "as an XYZ frame. This usually means the JDFTx output was missing "
            "or truncated -- check its RunStatus before exporting."
        )
    if include_forces and len(label.forces_ev_per_angstrom) != n_atoms:
        raise ValueError(
            f"Label {label.state_point_id}: {len(label.forces_ev_per_angstrom)} "
            f"force rows for {n_atoms} atoms. Refusing to write a frame whose "
            "forces and positions disagree."
        )
    if label.n_electrons is None:
        raise ValueError(
            f"Label {label.state_point_id} has no n_electrons. CP-MACE requires "
            "the `electron=` tag; a grand-canonical JDFTx run reports it as "
            "nElectrons. Re-harvest with data.harvest._parse_n_electrons or "
            "exclude this label."
        )

    lattice = " ".join(f"{c:.8f}" for row in label.cell_angstrom for c in row)
    properties = ("species:S:1:pos:R:3:forces:R:3" if include_forces
                  else "species:S:1:pos:R:3")

    info = [
        f'Lattice="{lattice}"',
        f"Properties={properties}",
        f"energy={label.energy_ev:.10f}",
        f"electron={label.n_electrons:.6f}",
        f"potential={label.mu_ev:.6f}",
        'pbc="T T T"',
    ]
    if extra_info:
        for key, value in sorted(extra_info.items()):
            info.append(f"{key}={value}")

    lines = [str(n_atoms), " ".join(info)]
    for i in range(n_atoms):
        x, y, z = label.positions_angstrom[i]
        row = f"{label.species[i]:<3s} {x:16.8f} {y:16.8f} {z:16.8f}"
        if include_forces:
            fx, fy, fz = label.forces_ev_per_angstrom[i]
            row += f" {fx:16.8f} {fy:16.8f} {fz:16.8f}"
        lines.append(row)
    return "\n".join(lines) + "\n"


def append_cp_mace_frame(path, label: CPDFTLabel,
                         include_forces: bool = True,
                         extra_info: Optional[dict] = None) -> None:
    """Append one frame. Lets a long campaign stream labels to disk instead of
    holding a whole dataset in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(_format_frame(label, include_forces, extra_info))


def write_cp_mace_xyz(path, labels: Iterable[CPDFTLabel],
                      include_forces: bool = True,
                      mu_tol_ev: float = 0.05,
                      skip_unusable: bool = True,
                      strict: bool = False) -> dict:
    """Write a CP-MACE training set.

    By default, labels that are unconverged or that missed their target mu by
    more than `mu_tol_ev` are SKIPPED, and the count is returned. Training a
    potential-aware model on labels that never reached their target potential
    teaches it the wrong mapping, and the failure is invisible afterwards.

    `strict=True` raises on the first unusable label instead. Use it when the
    caller has already filtered and any rejection indicates a logic error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped_unconverged = skipped_off_target = skipped_malformed = 0
    skipped_unknown_etype = 0
    reasons: List[str] = []
    etypes_seen = set()

    with open(path, "w") as fh:
        for label in labels:
            if skip_unusable:
                if label.etype is None:
                    # A label whose energy type is unknown may be a grand
                    # potential G or a Helmholtz F; those differ by mu*N, which
                    # is ~1683 eV in a shipped pymatgen fixture. Training on a
                    # mixture is silently, catastrophically wrong.
                    skipped_unknown_etype += 1
                    reasons.append(
                        f"{label.state_point_id}: etype unknown (G vs F "
                        "indeterminate); re-harvest with a pymatgen that "
                        "exposes JDFTXOutfile.etype")
                    if strict:
                        raise ValueError(reasons[-1])
                    continue
                etypes_seen.add(label.etype)
                if not label.converged:
                    skipped_unconverged += 1
                    if strict:
                        raise ValueError(f"{label.state_point_id}: not converged.")
                    continue
                err = label.mu_error_ev
                if err is not None and abs(err) > mu_tol_ev:
                    skipped_off_target += 1
                    reasons.append(
                        f"{label.state_point_id}: mu off target by {err:+.4f} eV")
                    if strict:
                        raise ValueError(reasons[-1])
                    continue
            try:
                fh.write(_format_frame(label, include_forces))
                written += 1
            except ValueError as exc:
                skipped_malformed += 1
                reasons.append(str(exc))
                if strict:
                    raise

    if len(etypes_seen) > 1:
        raise ValueError(
            f"Refusing to emit a mixed-energy-type dataset: saw {sorted(etypes_seen)}. "
            "Grand potential G and Helmholtz F differ by mu*N. Convert with "
            "CPDFTLabel.helmholtz_f_ev, or filter to one etype, before training."
        )

    return {
        "path": str(path),
        "etype": (etypes_seen.pop() if len(etypes_seen) == 1 else None),
        "n_skipped_unknown_etype": skipped_unknown_etype,
        "n_written": written,
        "n_skipped_unconverged": skipped_unconverged,
        "n_skipped_off_target_mu": skipped_off_target,
        "n_skipped_malformed": skipped_malformed,
        "mu_tol_ev": mu_tol_ev,
        "reasons": reasons[:20],
    }


def split_train_valid(labels: Sequence[CPDFTLabel], valid_fraction: float = 0.1,
                      seed: int = 0, by_state_point: bool = True):
    """Split labels into train/valid.

    `by_state_point=True` (default) keeps every frame from one state point on
    the same side of the split. Frames from a single trajectory are highly
    correlated, so a naive random split leaks and reports a validation error
    several times better than the model's real transferability.
    """
    import random

    labels = list(labels)
    rng = random.Random(seed)
    if not by_state_point:
        shuffled = labels[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1.0 - valid_fraction))
        return shuffled[:cut], shuffled[cut:]

    groups: dict = {}
    for label in labels:
        groups.setdefault(label.state_point_id, []).append(label)
    keys = sorted(groups)
    rng.shuffle(keys)
    cut = max(1, int(len(keys) * (1.0 - valid_fraction))) if keys else 0
    train_keys, valid_keys = set(keys[:cut]), set(keys[cut:])
    train = [l for k in train_keys for l in groups[k]]
    valid = [l for k in valid_keys for l in groups[k]]
    return train, valid
