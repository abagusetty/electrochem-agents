# Scientific Proposal — Agentic Constant-Potential Electrocatalysis

**Working title:** *Explicit-solvent grand-canonical electrocatalysis under agentic
control: Fermi-level uncertainty as an acquisition signal for CO dimerization at Cu*

**Scope of this document:** the scientific case — why this work, why now, what is
actually new, and who should care. Execution detail (architecture, phases, gates,
metrics) lives in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); module-level implementation
status lives in [`ROADMAP.md`](ROADMAP.md).

**Anchor paper:** Sahoo, Maraschin, Varley, Levine, **Ulissi**, Zitnick, Takemura,
Gauthier, Govindarajan, Shuaibi — *Insights into CO dimerization at electrified Cu
interfaces from large-scale machine learning simulations.* arXiv:2509.17862
(v1 2025-09-22, v2 2026-06-08), DOI 10.48550/arXiv.2509.17862. `[V-1]`

> Earlier drafts in this project mis-titled 2509.17862 as "The Open Catalyst 2025
> Dataset and Models for Solid–Liquid Interfaces." Wrong. It *introduces* OC25, but
> its subject is the Cu electrocatalysis study. Cite it accordingly.

> **Provenance convention.** Every non-trivial claim is tagged: `[V-n]` verified
> against a live source in the ledger (`RESEARCH_PLAN.md` §9); `[C]` verified by
> reading source code; `[P]` project capability asserted by the team;
> `[A]` assumption or design choice, not a fact; `[?]`
> unverified — must not enter a manuscript. This exists because an earlier session
> in this project produced a report in which 100% of citations were unverifiable.
> Do not relax it.

---

## 1. Thesis

Simulating an electrified interface honestly requires three things at once:

1. **The right ensemble.** An electrode is a potentiostat, not a fixed-charge
   capacitor. The physical ensemble is grand-canonical in electrons at fixed potential.
2. **The right statistics.** Interfacial barriers need nanoseconds of explicit-solvent
   enhanced sampling, not a handful of NEB images.
3. **The right scope.** Mechanism depends jointly on facet, potential, cation, and
   coverage. One point in that space is an anecdote.

The anchor paper achieves (2) and much of (3) — >800-atom cells, up to 7 ns, "the
largest explicit-solvent CO dimerization study to date" — by trading away (1). Its MD
is **constant-charge**, made approximately constant-potential by choosing cells large
enough that reaction-induced work-function shifts stay small. `[V-1]` Its Discussion
names the fix: couple OC25 models to grand-canonical approaches, or train models that
directly predict the work function. `[V-1]`

**The claim of this effort:** all three are simultaneously reachable, and the enabling
ingredient is not a larger model. It is a principled answer to *where to spend
grand-canonical DFT*. Explicit-solvent GC-DFT is affordable only if something decides,
adaptively and defensibly, which few hundred state points out of a combinatorial space
deserve labels. That decision problem is the scientific contribution — and it is what
separates agentic *science* from a workflow with an LLM stapled to it.

---

## 2. The gap, stated precisely

Two literatures have each solved half the problem, in opposite halves.

|  | **Constant charge** | **Constant potential** |
|---|---|---|
| **Implicit solvent** | routine | **BEAST DB** — 20,000+ grand-canonical JDFTx surface calculations, consistent parameters, open `[V-3]` |
| **Explicit solvent** | **OC25** — 7.8M DFT single points, 1.5M interface configurations, 88 elements `[V-1]` | *empty* |

The empty cell is empty for one reason: explicit-solvent grand-canonical DFT is
expensive. That is not an argument for avoiding it. It is the argument for why an
**acquisition policy** is the scientific contribution rather than an engineering detail.

Two further facts sharpen the position:

- BEAST DB includes **Sundararaman**, JDFTx's lead author `[V-3]` — so the
  grand-canonical protocol conventions (CANDLE solvation, potential referencing) have a
  community standard to follow rather than reinvent. `[V-4]`
- The anchor paper is a **Meta FAIR Chemistry + LLNL + Texas Tech** collaboration
  `[V-1]` — so a DOE-lab entrant with GC-DFT capability is joining an established
  collaboration pattern, not proposing an unusual one.

### 2.1 What the anchor paper establishes `[V-1]`

- OC25-trained eSEN reproduces metal–water interfacial structure where foundation
  models fail — MACE-MP-0 fails *qualitatively* at Rh(111)/water (unphysical O-density
  spike near z≈1 Å).
- CO dimerization on Cu(100) is weakly sensitive to surface charge and cation identity;
  appreciable stabilization only at the most negative charge densities. Water
  reorientation screens the field.
- Stepped Cu(310) opens a more favorable pathway at modest reducing potentials —
  under-coordinated sites are the real C–C coupling centers.

### 2.2 The four open items the authors name `[V-1]`

1. Constant-charge, not constant-potential — approximate, and area-dependent.
2. *"Couple OC25 models with grand-canonical approaches or develop models that directly
   predict the work function."* An explicit invitation.
3. Local message-passing GNNs with finite cutoffs — no explicit long-range
   electrostatics, which is precisely what orders ions in the double layer.
4. Universal foundation models are not yet suited to electrified solid–liquid interfaces.

Item 2 is what this proposal attacks. Item 3 is a scoped, acknowledged limitation
(`RESEARCH_PLAN.md` G6).

---

## 3. What is actually new

Stated adversarially, because the failure mode here is claiming novelty that a
reviewer can dismantle in one search.

**Already claimed — cite, do not claim.** A 105-agent adversarially-verified
literature sweep (2026-08-20) found this space far more occupied than earlier
drafts assumed. `[V-7]`

- *Grand-canonical DFT.* JDFTx `target-mu` since 2017; grand-canonical
  Nose–Hoover on electron number since Bonnet 2012.
- *Potential-conditioned MLIPs.* **Four independent lines**, not one: CP-MACE
  (JCTC 2025, Ni–N–C and Au — not Cu/water) `[V-2]`; **PE-MACE / EEP-MLFF
  (arXiv:2604.07322, April 2026)**, scalar potential embedded into initial node
  features; **TRECI (arXiv:2511.19338)**, seven discrete kernel readout heads on
  frozen MACE features, one per bias from −0.50 to −2.00 V vs SHE — and the
  strongest data-efficiency prior art; plus 10.1021/acs.jctc.5c01381.
- *Constant-potential enhanced sampling.* CP-MACE already ships a
  constant-potential metadynamics driver.
- *Grand-canonical DFT databases.* BEAST DB, implicit solvent. `[V-3]`
- *Agentic computational chemistry — including electrocatalysis.* ChemGraph
  (arXiv:2506.06363); **Catalyst-Agent (arXiv:2603.01311), already doing
  closed-loop autonomous ORR/NRR/CO2RR screening**; TritonDFT; LARA; AutoDFT.
- *Multi-agent orchestration on a DOE leadership machine.* **arXiv:2604.07681
  (April 2026) — an Argonne planner–executor framework running gpt-oss-120b +
  MCP + Parsl on Aurora.** "LLM agents drive simulations on Aurora" is not a
  paper.
- *Abductive/inductive exploration split.* A-Lab GPSS (arXiv:2604.11957).

**No single layer of this stack survives review as novel on its own.** The
defensible unit is the composition, and the two things inside it that remain
genuinely unclaimed.

**Unclaimed #1 — a work-function/Fermi-level head on the OC25/UMA/eSEN
foundation-model line.** This is the strongest item in the proposal, because
FAIR Chemistry states it as an open direction *in their own paper*, verbatim
(§5). Everyone building potential-conditioned MLIPs today trains a bespoke
model; nobody has put the head on the foundation-model line.

**Unclaimed #2 — JDFTx `target-mu` as the label source, at Aurora throughput.**
The capability half matters as much as the idea: JDFTx is ported to SYCL and runs
GPU-native on Ponte Vecchio `[P]`, while upstream has no Intel-GPU backend `[V-7]`. Every competing
constant-potential MLIP uses VASP with explicit ions or a double-reference
counter-charge scheme. Nobody has used implicit-electrolyte grand-canonical
JDFTx labels for this. Note the honest qualification: JDFTx hard-gates
`target-mu` on an electrolyte (`fluid-cation`/`fluid-anion` required, parser
enforces it fatally), so these labels are **explicit water, implicit ions** —
which must be disclosed, and which is precisely what makes the calibration
question below interesting.

**The defensible contribution — σ_µ as an acquisition signal.**

Reading CP-MACE's metadynamics driver revealed that it already runs a two-model
committee whose `AverageForceCalculator` reports not only force standard deviation but
**chemical-potential variation across the committee**. `[C]`

That quantity — call it **σ_µ, disagreement in the predicted electrode potential** —
has a property nothing in the constant-charge literature can have: it is disagreement
about the *electronic boundary condition itself*, not about a force component. A
constant-charge committee cannot produce it, because there is no µ to disagree about.

The proposal is to make σ_µ the trigger for spending grand-canonical DFT:

> When the committee disagrees about what potential the system is at, that state point
> has earned a GC-DFT label. When it merely disagrees about forces, cheaper remedies
> apply.

This is CP-native, physically interpretable, and — as far as verified — unclaimed.
It is also the one form of agency the crowded agentic literature has not taken:
those systems have agents control **job submission**; this has agents control
**the physics** — µ setpoints, bias parameters, convergence-triggered stopping.
Frame it that way or not at all.
It also comes with its own falsification test built in: an ablation against a
force-uncertainty-only trigger. If σ_µ carries no information beyond σ_F, the
experiment says so and the paper is retitled. (`RESEARCH_PLAN.md` §3.3.)

**Also new, and lower-risk:** the explicit-solvent grand-canonical treatment of the
specific Cu(100)/Cu(310) CO-dimerization system, with the constant-charge-vs-constant-
potential discrepancy *mapped as a function of cell size* rather than assumed small.

**Framing discipline:** the headline metric is *cost to reproduce published values*,
never "we built an agent." If adaptive-OPES prior art surfaces, cite it head-on and
differentiate on the σ_µ audit policy.

---

## 4. Contributions

| | Contribution | Why it is not redundant |
|---|---|---|
| **C1** | First explicit-solvent, grand-canonical DFT reference set for Cu(100)/Cu(310) CO dimerization; ΔΔG = ΔG_CP − ΔG_CC mapped vs cell size, facet, cation | fills the empty cell in §2 |
| **C2** | Potential-aware MLIP for Cu/water, two routes benchmarked head-to-head: µ-head on eSEN-OC25 vs FermiMACE retrained on Cu | CP-MACE is Ni–N–C/Au `[V-2]`; anchor paper requests exactly this `[V-1]` |
| **C3** | **σ_µ-triggered acquisition** — committee disagreement in the Fermi level schedules GC-DFT; plus mid-run OPES hyperparameter control | CP-native signal, with a built-in ablation |
| **C4** | Open artifacts: the CP label set, the µ-heads, and the anchor paper's reported values as a scored benchmark task | complements OC25; does not compete with it |

**C3 is the headline. C1 and C2 make it credible rather than a demo. C4 is the reason
FAIR Chemistry would want in.**

C2 deserves one note: the two routes are not a hedge, they are an experiment. *Does a
potential-native model trained on less relevant data beat an interface-specialized
model with a potential head bolted on?* Nobody has answered that, and answering it
converts a private indecision into a public result.

---

## 5. Why FAIR Chemistry / Zachary Ulissi should want in

Ulissi is an author on the anchor paper. `[V-1]` This is not cold outreach — it is a
continuation of a paper he wrote.

**Do not pitch:** *"here is an idea about your model."*
**Pitch:** *"your Discussion names two open items; we have a DOE-lab grand-canonical
capability and a first result on one of them."*

1. **It answers a stated open item verbatim.** From the anchor paper's *Outlook
   and future directions*, Ulissi a coauthor `[V-7]`:

   > "An important quantity that is currently not explored in this work is the
   > interface workfunction… it would be highly desirable to have ML models that
   > can predict the interface workfunctions directly. Access to the interface
   > workfunction during the simulation can also enable constant potential
   > (grand-canonical) simulations… Developing models that can accurately predict
   > both the Fermi level and the vacuum potential, and in turn the interfacial
   > workfunction, is an exciting direction for future research."

   A µ-head on their architecture is literally that. This quote belongs in the
   first paragraph of any outreach — it is FAIR publicly posting the vacancy.
2. **It supplies labels they cannot cheaply make.** Explicit-solvent GC-DFT sits outside
   the pipeline that produced OC25. The community has GC-DFT at 20k scale in *implicit*
   solvent `[V-3]`; nobody has the explicit cell. Complementary capability, not
   duplicated effort.
3. **It probes UMA where UMA is weak, constructively.** Their own numbers show UMA-OC20
   trailing eSEN-OC25 interfacially, and UMA-S-ft(OC25) still trailing on strict
   splits. `[V-1]` *Does a µ-head transfer across UMA task heads?* is a foundation-model
   question — their research question, not ours.
4. **It ships in the format their ecosystem consumes.** Dataset + checkpoints + a scored
   benchmark task. Leaderboard-shaped.
5. **The acquisition layer is orthogonal to what FAIR builds.** Their strength is models
   and data at scale. A σ_µ-triggered DFT-in-the-loop policy is a different
   contribution, not a competing one.
6. **The complementary asset is concrete, not nominal.** JDFTx is ported to
   SYCL and runs GPU-native on Aurora's Ponte Vecchio GPUs `[P]`; upstream JDFTx
   has no Intel-GPU backend of any kind `[V-7]`. FAIR has models and data at
   scale; what they cannot buy is grand-canonical DFT throughput at leadership
   scale. That is exactly what this side brings, and it is why the collaboration
   is an exchange rather than a request.

**The ask, bounded:** co-authorship on the label-set/benchmark release, plus advisory
input on the µ-head architecture. Low cost, high visibility, no open-ended commitment.

**Secondary channels:** Govindarajan and Varley (LLNL) — DOE-lab peers already
collaborating on this exact system, and Govindarajan additionally authored the field's
methodological-guardrails paper, whose checks this project encodes as automated
validation. Gauthier (Texas Tech). Separately, the Liu group, whom the CP-MACE
licensing question (`RESEARCH_PLAN.md` G8) gives a natural, non-presumptuous reason to
contact.

**Timing.** Credibility comes from reproducing the anchor paper's numbers first
(Phase 1). But the people best positioned to close this gap are the people who named
it — so do not wait past Phase 3. See `RESEARCH_PLAN.md` G10.

---

## 6. Deliverables and venues

Ranked by defensibility, after the literature sweep `[V-7]`:

- **Paper 1 — a work-function/Fermi head for foundation atomic models, trained
  on grand-canonical DFT.** C1 + C2, reframed. The only framing with a
  primary-source-verified statement from the target collaborator that the work
  is wanted and not being done. Dataset + model, not a wrapper.
  *JACS* / *Nature Catalysis* / *npj Comput. Mater.*
- **Paper 2 — cross-engine calibration of constant-potential methods.** *New,
  promoted on the strength of the sweep.* Four incompatible conventions now
  coexist — JDFTx `target-mu` (implicit electrolyte), VASP double-reference/FCP
  (TRECI), explicit-ion constant charge (OC25), potential-conditioned MLIPs —
  with no cross-walk. Quantify how they disagree on the same reaction at the
  same potential; publish the transfer functions and the G-vs-F/µN bookkeeping
  needed to mix them in one training set. Unglamorous, lowest preemption risk,
  citable by everyone. *JCTC* / *Digital Discovery*.
- **Paper 3 — agent-in-the-loop constant-potential campaigns.** C3 + C4. Ranked
  last on its own merits: Catalyst-Agent, TritonDFT, LARA and arXiv:2604.07681
  already occupy the surrounding space, so this works as a *component* of
  Paper 1 or 2 rather than as a headline. *Nature Machine Intelligence* /
  *Digital Discovery*, or a NeurIPS AI4Science track.
- **Paper 0 — the JDFTx SYCL port on Aurora.** *Not opportunistic: the port
  exists* `[P]`. Upstream JDFTx has no SYCL/oneAPI/HIP backend at all — verified
  against `CMakeLists.txt`, the tree, and the GitHub language API `[V-7]` — so a
  production grand-canonical DFT code running GPU-native on Ponte Vecchio, with
  performance data at leadership scale, is a contribution in its own right.
  SC / ISC / IXPUG. It is also the enabling moat for Papers 1–3: GC-DFT
  throughput is the scarce input, and nobody else has this path to it.
- **Artifacts:** the grand-canonical explicit-solvent label set (the reusable asset —
  OC25 is constant-charge at scale; this is constant-potential at depth), the trained
  µ-heads, and the reproduction table as a scored benchmark.
- **Negative results are deliverables.** If ΔΔG is small at all cell sizes, that
  validates a large body of constant-charge work and is worth reporting. If the agentic
  loop does not beat a force-uncertainty baseline, that is a finding about uncertainty
  signals and is worth reporting. A plan whose every outcome is publishable is a
  well-posed plan.

---

## 7. Open scientific questions

1. Which JDFTx solvation setting is defensible for Cu/water at negative potential, and
   what does one GC single point cost at the required cell size?
2. What is the data-efficiency curve for a µ-head — hundreds of labels, or thousands?
   This determines whether C2 is a paper or a footnote.
3. Does the σ_µ audit policy generalize across facets, or overfit to Cu?
4. Does σ_µ decompose into interpretable contributions (double-layer structure vs
   adsorbate charge transfer)? If so, C3 becomes a physical-insight paper, not only a
   cost paper.
5. Is OMol25/UMA anything here beyond a baseline and a transferability probe?
6. What exactly is the pre-registered grid baseline? The comparison is meaningless
   until fixed in advance.

---

## Appendix — document map

| Document | Role |
|---|---|
| `PROPOSAL.md` | this file — scientific case, novelty, collaboration |
| `RESEARCH_PLAN.md` | architecture, phases, gates, metrics, source ledger |
| `ROADMAP.md` | module-by-module implementation status |
| `archive/session-transcript-2026-08.md` | raw 42-turn design conversation; provenance only, **not** a specification |
