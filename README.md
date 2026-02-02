# RIBDP — Robust Intent-Based Belief Dynamic Planner

[![CI](https://img.shields.io/badge/ci-none-lightgrey)](https://github.com/<USER>/<REPO>/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Torch optional](https://img.shields.io/badge/torch-optional-orange.svg)]()

> Robust planning under adversarial observation manipulation. Not a toy. A concise, reproducible reference implementation and evaluation pipeline for minimax belief-space planning under KL-bounded adversaries.

---

## Quick elevator

- **What:** A belief-space planner that computes policies robust to intentional observation manipulation constrained by a KL divergence budget `ε`.
- **Why:** Many deployed systems assume honest sensing. RIBDP assumes an adversary and produces plans that are provably robust against the worst-case within a bounded information budget.
- **Status:** Reference code + experiments. Paper **not included** in this repo (unpublished). See `results/` for example outputs.

---

## Badges — replace `<USER>/<REPO>`
These badges are placeholders. Replace the links with your repo paths, CI workflow and other badges you actually enable.

```md
[![CI](https://img.shields.io/github/actions/workflow/status/<USER>/<REPO>/ci.yml)](https://github.com/<USER>/<REPO>/actions)
[![Release](https://img.shields.io/github/v/release/<USER>/<REPO>)](https://github.com/<USER>/<REPO>/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
Repo layout (concise)
.
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ main.py                         # orchestrator (fast/exact/torch flags)
├─ ribdp_core.py                   # algorithm + solver
├─ run_performance_evaluation.py
├─ run_adversary_analysis.py
├─ run_ablation_study.py
├─ run_scenario_simulation.py
└─ results/                        # committed final figures & CSVs
How to run (copy-paste)
Install:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run everything (fast mode, cached):

python main.py
Run exact (no caches) for validation:

python main.py --no-fast
Run a single experiment and try numeric acceleration:

python main.py --run adversary --use-torch --device cpu
# or --device cuda if you have a capable GPU and installed torch with CUDA
System design (short, technical)
Maintain a belief b over latent discrete states S.

At each decision step choose action a to minimize expected loss under the worst-case observation model q(x) satisfying KL(q || p) ≤ ε, where p(x) is the nominal sensor model.

Solve the inner adversary optimization in closed form (exponential tilt), embed it into a minimax Bellman operator, and compute value functions by value iteration over a discretized belief grid.

Provide reproducible experiment scripts for performance sweeps, ablation and adversary characterization.

Math (readable, exact)
Notation
S — finite latent states (|S|)

A — actions

X — observation space (discrete or sampled continuous)

b ∈ Δ^{|S|} — belief (probability vector)

p(x | s) — nominal observation likelihood

L(a, s) — per-step defender loss

γ — discount factor

ε — KL budget (adversary constraint)

ℓ(x) — scalar functional of observation x (typically posterior expected cost after seeing x)

Inner adversary problem (single-step)
For a given scalar functional ℓ(x) the adversary solves:

maximize    E_{x ~ q}[ ℓ(x) ]
subject to  KL(q || p) ≤ ε
           q is a distribution over X
Form the Lagrangian with multiplier λ > 0:

L(q, λ) = Σ_x q(x) ℓ(x) + λ ( ε - Σ_x q(x) log(q(x)/p(x)) )
Stationarity w.r.t. q(x) yields:

q_λ(x) ∝ p(x) * exp( ℓ(x) / λ )
So the worst-case adversary is an exponential tilt of p. Choose λ so that

KL(q_λ || p) = ε
(i.e., solve a single scalar equation — we use a robust root-finder in the implementation.)

Robust Bellman (minimax value)
Let V(b) be the robust value function over beliefs. The minimax Bellman equation used is:

V(b) = max_a  min_{q: KL(q||p) ≤ ε}
         { -E_{s~b}[ L(a, s) ]  + γ * E_{x~q}[ V( b'(b,a,x) ) ] }
b'(b, a, x) is the Bayesian posterior over states after action a and observation x using the nominal p(x|s) in the Bayes update.

Implementation notes:

inner min is replaced by q_λ from above

for continuous X we approximate expectations via importance sampling:

sample x_i ~ p, compute weights w_i ∝ exp(ℓ(x_i)/λ) and normalize

estimates converge as O(1/√N) but suffer in high-dim X

Implementation details (what the code does)
ribdp_core.py:

exponential_tilt(p, loss, ε) — robust adversary solver (numerically stable)

RIBDPSolver — value iteration over a discretized belief grid

caching options (fast=True by default) to avoid repeated transitions / nearest-neighbor lookups

main.py:

orchestrates experiments

flags: --fast / --no-fast, --use-torch, --device, --seed, --run


Current limitations — be blunt, so reviewers stop asking the same questions
This list is explicit so you can quote it in interviews and look smart.

Assumes reasonably accurate nominal models (p(x|s), transition P)

Why it matters: If p is wrong, the exponential-tilt adversary tilts the wrong distribution. The robust policy then optimizes for the wrong threat model.

Real-world fix: online calibration / posterior over models / ensembles.

Single-step KL budgets (local adversary)

Why it matters: Real attackers plan over many steps and allocate resources stealthily. Per-step ε ignores cross-step tradeoffs.

Real-world fix: cumulative KL constraints or attacker-cost planning (MPEC-style).

Belief discretization / projection bias

Why it matters: Nearest-neighbor projection of continuous posteriors onto a grid introduces bias that can change policies, especially with coarse grids.

Real-world fix: particle filters, learned belief representations, or parametric belief compression.

Sample complexity for continuous / high-dim observations

Why it matters: Importance-sampling weights collapse in high-dimensional X, yielding high variance estimates of expected future value.

Real-world fix: adaptive sampling, normalizing flows to model tilted q, control variates.

Scalability & runtime

Why it matters: Exact value iteration over belief grids and per-step root-finding is computationally expensive; not suitable for sub-second decision loops.

Real-world fix: precompute adversarial shifts, use MPC with short horizons, or move to learned robust policies (actor-critic over belief embeddings).

Evaluation realism

Why it matters: Current scenarios use surrogate classifiers and synthetic perturbations — not the same as measured attacker telemetry.

Real-world fix: red-team datasets, replayed telemetry, and end-to-end testing.

Bottom line: this codebase is a reproducible, research-grade reference. It is not a drop-in industrial solution. That’s intentional — it’s the right interim step between math and deployment.

What’s improved in upcoming versions (roadmap)
Planned near-term improvements (v2):

cumulative/adaptive adversary budgets; parametric tilted-q approximations

belief compression and robust actor-critic implementation (scales to larger S)

realistic red-team dataset integration + CI tests for regression

optional torch backend for numerical kernels and a production mode (precomputed shifts & lookup)

Contribution & code hygiene
Keep PRs small. If you touch ribdp_core.py, include a unit test or a quick script in scripts/ that demonstrates identical output (or documents the intended deviation).

fast=True is the default for day-to-day experiments. Use --no-fast to validate exact behavior.

If you add dependencies, update requirements.txt and add a one-line explanation in README.md.

Citation / acknowledgment
This implementation is intended for research & demonstration. If you use RIBDP in publications or products, please cite the repository (commit hash) and add a short acknowledgement in your paper or documentation.

Contact / issues
Open an issue with:

the experiment command you ran

the commit hash

minimal logs and your requirements.txt

We deal with broken reproducibility quickly. We do not accept tickets titled “it does not work lol”.

Quick reference (commands)
# run default experiments (fast cache on)
python main.py

# run exact solver for debugging/validation
python main.py --no-fast

# run adversary analysis with torch numeric kernels on CPU
python main.py --run adversary --use-torch --device cpu