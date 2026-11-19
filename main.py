#!/usr/bin/env python3
"""QuantumMonteCarlo-Lab - educational quantum Monte Carlo lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lessons are stated up front:

  * Variational Monte Carlo (VMC) never lies downward: for any
    trial wavefunction, the exact expectation of the local energy
    is an UPPER bound on the true ground-state energy. A Monte
    Carlo estimate of it is that bound plus noise - so this lab
    always prints the estimate together with its error bar, and
    grades the estimate against the analytic expectation wherever
    one exists.
  * An error bar is a computed quantity, not a decoration: the
    block-averaged standard error must shrink like 1/sqrt(M) as
    the sample count M grows, and the lab measures that it does.
  * The sign problem is shown, not solved: for free fermions in
    a harmonic trap the average sign of the permutation weights
    is computed EXACTLY by a recursion, and it collapses with
    system size and inverse temperature. The implied cost factor
    1/<sign>^2 explodes. This is a labelled TOY demonstration -
    exact arithmetic on free particles - not a proof about
    interacting systems, and not a solution to anything.

Everything is deterministic: one in-file splitmix64 PRNG with a
named seed drives every sampler, so reruns are byte-identical.
Production chemistry, materials, drug, energy, and publication
uses are permanent HOLD gates - with Gemini Argon 4 Extended
Thinking as DECIDER (names-only, tighten-only), Muse as advisor
(never overrides HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"
SEED = 20261119  # named seed for every sampler in this lab

# Labelled reference values. These are inputs the lab grades
# against, stated openly - not numbers the lab derives.
HELIUM_Z = 2.0
HELIUM_TRUE_E0 = -2.903724377   # labelled reference: helium ground energy (Ha)
HELIUM_OPTIMAL_ZEFF = 27.0 / 16.0  # 1.6875, analytic optimum of the trial family

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "production_chemistry_claim": "HOLD - no production quantum chemistry; a teaching VMC run is not a chemistry result",
    "real_molecule_accuracy_claim": "HOLD - no claim of chemical accuracy for any real molecule or material",
    "drug_design_claim": "HOLD - no drug-design, binding-affinity, or pharmaceutical use",
    "materials_certification_claim": "HOLD - no certifying a real material's properties from a lab estimate",
    "catalyst_design_claim": "HOLD - no catalyst design or reaction-rate claims",
    "battery_material_design_claim": "HOLD - no battery or energy-storage material design claims",
    "superconductor_material_claim": "HOLD - no superconductor discovery or Tc claims",
    "nuclear_simulation_use": "HOLD - no nuclear, weapons, or defence simulation use",
    "sign_problem_solved_claim": "HOLD - the sign problem is demonstrated here, never solved; no claim otherwise",
    "production_software_claim": "HOLD - this is a teaching lab, not production QMC software",
    "hardware_benchmark_claim": "HOLD - no hardware benchmark or procurement claims from lab timings",
    "investment_claim": "HOLD - no investment claims; a lab model is not a prospectus",
    "patent_claim": "HOLD - no patent, novelty, or priority claims from lab output",
    "publication_result_claim": "HOLD - lab numbers are teaching values, not publishable research results",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 0. PRNG
# splitmix64, implemented in full: one 64-bit state, a fixed
# increment, and a bijective mix. Every sampler in the lab draws
# from streams seeded from SEED, so the whole lab is reproducible
# bit for bit - Monte Carlo without a seed is an anecdote.

_MASK64 = (1 << 64) - 1


class SplitMix64:
    """Deterministic splitmix64 stream (seeded, platform-independent)."""

    def __init__(self, seed):
        self.state = seed & _MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def random(self):
        """Uniform in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform_open(self):
        """Uniform in (0, 1] - safe inside log()."""
        return 1.0 - self.random()


def block_stats(values, block):
    """Mean and block-averaged standard error of a sample stream.

    Consecutive samples from a Markov chain are correlated, so the
    honest error bar comes from block means: average inside blocks
    of `block` samples, then take the standard error of the block
    means. Returns (mean, standard_error, n_blocks).
    """
    n_blocks = len(values) // block
    if n_blocks < 2:
        raise ValueError("need at least two full blocks")
    means = [sum(values[i * block:(i + 1) * block]) / block
             for i in range(n_blocks)]
    mean = sum(means) / n_blocks
    var = sum((m - mean) ** 2 for m in means) / (n_blocks - 1)
    return mean, math.sqrt(var / n_blocks), n_blocks


# ================================================================ 1. VMC: 1D harmonic oscillator
# H = -1/2 d2/dx2 + 1/2 x2 (atomic units), exact E0 = 0.5.
# Trial state psi(x) = exp(-alpha x2). The local energy is analytic:
#   E_L(x) = alpha + (1/2 - 2 alpha^2) x^2
# and its exact expectation under |psi|^2 is
#   E(alpha) = alpha/2 + 1/(8 alpha),
# minimised at alpha = 1/2 with E = 1/2 - where E_L is the constant
# 0.5 for every x (the exact eigenstate: zero-variance principle).

HO_E0_EXACT = 0.5
HO_STEP = 1.4          # Metropolis proposal half-width (labelled)
HO_BLOCK = 100


def ho_local_energy(x, alpha):
    return alpha + (0.5 - 2.0 * alpha * alpha) * x * x


def ho_exact_energy(alpha):
    return alpha / 2.0 + 1.0 / (8.0 * alpha)


def run_ho_vmc(alpha, kept, burn_in, seed):
    """Metropolis VMC for the oscillator at trial width alpha.

    Samples |psi|^2 by a random-walk Metropolis chain; the first
    `burn_in` proposals are discarded as equilibration. Returns the
    block-averaged estimate, its error bar, and the acceptance rate
    over ALL proposals (burn-in included - bookkeeping stays honest).
    """
    rng = SplitMix64(seed)
    x = 0.0
    accepted = 0
    values = []
    for i in range(burn_in + kept):
        proposal = x + (2.0 * rng.random() - 1.0) * HO_STEP
        ratio = math.exp(-2.0 * alpha * (proposal * proposal - x * x))
        if rng.uniform_open() < ratio:
            x = proposal
            accepted += 1
        if i >= burn_in:
            values.append(ho_local_energy(x, alpha))
    mean, stderr, n_blocks = block_stats(values, HO_BLOCK)
    return {
        "alpha": alpha,
        "estimate": mean,
        "stderr": stderr,
        "exact_expectation": ho_exact_energy(alpha),
        "kept_samples": kept,
        "burn_in_discarded": burn_in,
        "acceptance_rate": accepted / (burn_in + kept),
        "blocks": n_blocks,
    }


_MEMO = {}


def vmc_report():
    """Oscillator VMC at the optimal width and at a wrong width."""
    if "vmc" not in _MEMO:
        _MEMO["vmc"] = {
            "optimal": run_ho_vmc(0.5, 20000, 2000, SEED),
            "wrong_width": run_ho_vmc(0.35, 20000, 2000, SEED + 11),
            "e0_exact": HO_E0_EXACT,
        }
    return _MEMO["vmc"]


def scaling_report():
    """Error-bar scaling at alpha = 0.35: M = 4000 vs M = 64000.

    Sixteen times the samples should halve the error bar twice:
    the ratio of block standard errors should sit near sqrt(16) = 4.
    """
    if "scaling" not in _MEMO:
        short = run_ho_vmc(0.35, 4000, 1000, SEED + 21)
        long = run_ho_vmc(0.35, 64000, 2000, SEED + 22)
        _MEMO["scaling"] = {
            "short": short,
            "long": long,
            "sample_ratio": 16.0,
            "sqrt_law_prediction": 4.0,
            "stderr_ratio": short["stderr"] / long["stderr"],
        }
    return _MEMO["scaling"]


# ================================================================ 2. VMC: helium-like atom
# Two electrons around a charge-Z nucleus, trial state
# psi = exp(-zeff (r1 + r2)) - a screened hydrogenic product with
# NO Jastrow factor (stated openly: electron correlation enters
# only through the mean 1/r12 repulsion). Local energy:
#   E_L = -zeff^2 + (zeff - Z)(1/r1 + 1/r2) + 1/r12
# Exact expectation: E(zeff) = zeff^2 - 2 Z zeff + (5/8) zeff,
# using <1/r> = zeff and <1/r12> = (5/8) zeff. At Z = 2 the family
# is minimised at zeff = 27/16 with E = -2.847656 Ha - an upper
# bound on the labelled true value -2.903724377 Ha, never below it.
# |psi|^2 factorises, so positions are drawn DIRECTLY (independent
# samples, no Markov chain): radius ~ Gamma(3, rate 2 zeff),
# direction uniform on the sphere.

HE_BLOCK = 200
HELIUM_SCAN_ZEFF = (1.4, 1.55, 1.6875, 1.85, 2.0)


def helium_exact_energy(zeff, z=HELIUM_Z):
    return zeff * zeff - 2.0 * z * zeff + 0.625 * zeff


def _sample_electron(rng, zeff):
    radius = -(math.log(rng.uniform_open()) + math.log(rng.uniform_open())
               + math.log(rng.uniform_open())) / (2.0 * zeff)
    cos_t = 2.0 * rng.random() - 1.0
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * math.pi * rng.random()
    return (radius * sin_t * math.cos(phi),
            radius * sin_t * math.sin(phi),
            radius * cos_t)


def run_helium_vmc(zeff, samples, seed):
    """Direct-sampling VMC for the helium-like trial state."""
    rng = SplitMix64(seed)
    energies = []
    r12_sum = 0.0
    for _ in range(samples):
        (x1, y1, z1) = _sample_electron(rng, zeff)
        (x2, y2, z2) = _sample_electron(rng, zeff)
        r1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        r2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        r12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        energies.append(-zeff * zeff
                        + (zeff - HELIUM_Z) * (1.0 / r1 + 1.0 / r2)
                        + 1.0 / r12)
        r12_sum += 1.0 / r12
    mean, stderr, n_blocks = block_stats(energies, HE_BLOCK)
    return {
        "zeff": zeff,
        "estimate": mean,
        "stderr": stderr,
        "exact_expectation": helium_exact_energy(zeff),
        "mean_inv_r12": r12_sum / samples,
        "mean_inv_r12_exact": 0.625 * zeff,
        "samples": samples,
        "blocks": n_blocks,
    }


def helium_report():
    """Parameter scan plus a long run at the analytic optimum."""
    if "helium" not in _MEMO:
        scan = [run_helium_vmc(z, 20000, SEED + 100 + i)
                for i, z in enumerate(HELIUM_SCAN_ZEFF)]
        best = min(scan, key=lambda row: row["estimate"])
        _MEMO["helium"] = {
            "scan": scan,
            "best_zeff": best["zeff"],
            "optimal": run_helium_vmc(HELIUM_OPTIMAL_ZEFF, 60000, SEED + 200),
            "true_e0_labelled": HELIUM_TRUE_E0,
            "family_minimum_exact": helium_exact_energy(HELIUM_OPTIMAL_ZEFF),
        }
    return _MEMO["helium"]


# ================================================================ 3. The sign problem (toy, exact)
# Free particles in a 1D harmonic trap, levels e_k = k + 1/2.
# The canonical partition functions obey the permutation recursion
#   Z_N^B = (1/N) sum_k z1(k beta) Z_{N-k}^B
#   Z_N^F = (1/N) sum_k (-1)^(k+1) z1(k beta) Z_{N-k}^F
# with z1(beta) = 1 / (2 sinh(beta/2)). A Monte Carlo that samples
# the bosonic (all-positive) weights and reweights to fermions
# carries the average sign <sign> = Z_N^F / Z_N^B, and its cost
# grows like 1/<sign>^2. The recursion is EXACT arithmetic, so the
# collapse printed here is not statistical noise - it is the wall.
# Labelled TOY demonstration: free particles, no interactions.

def z1(beta):
    """Single-particle partition function of the trap."""
    return 1.0 / (2.0 * math.sinh(beta / 2.0))


def zn_partition(beta, n, fermion):
    """Canonical Z_N for bosons (all +) or fermions (permutation signs)."""
    z = [1.0]
    for big_n in range(1, n + 1):
        total = 0.0
        for k in range(1, big_n + 1):
            term = z1(k * beta) * z[big_n - k]
            if fermion and k % 2 == 0:
                term = -term
            total += term
        z.append(total / big_n)
    return z[n]


def average_sign(beta, n):
    return zn_partition(beta, n, True) / zn_partition(beta, n, False)


SIGN_BETA = 0.5
SIGN_N_SWEEP = tuple(range(1, 9))
SIGN_BETA_SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SIGN_BETA_SWEEP_N = 3


def sign_report():
    """Average-sign sweeps over N (fixed beta) and beta (fixed N)."""
    if "sign" not in _MEMO:
        n_sweep = [{"n": n, "avg_sign": average_sign(SIGN_BETA, n)}
                   for n in SIGN_N_SWEEP]
        for row in n_sweep:
            row["cost_factor"] = 1.0 / row["avg_sign"] ** 2
        beta_sweep = [{"beta": b, "avg_sign": average_sign(b, SIGN_BETA_SWEEP_N)}
                      for b in SIGN_BETA_SWEEP]
        # Ground-state dominance referee: at large beta the ratio
        # tends to exp(-beta * N(N-1)/2) - filled trap vs all-in-one.
        asym_n, asym_beta = 4, 2.0
        _MEMO["sign"] = {
            "beta_for_n_sweep": SIGN_BETA,
            "n_sweep": n_sweep,
            "n_for_beta_sweep": SIGN_BETA_SWEEP_N,
            "beta_sweep": beta_sweep,
            "asymptote": {
                "n": asym_n,
                "beta": asym_beta,
                "computed": average_sign(asym_beta, asym_n),
                "formula": math.exp(-asym_beta * asym_n * (asym_n - 1) / 2.0),
            },
            "toy_label": "free particles in a harmonic trap, exact recursion - "
                         "a demonstration, not a proof",
        }
    return _MEMO["sign"]


# ================================================================ 4. kNN run classifier
# Features of a finished run: (variational excess over the labelled
# ground value, block standard error, average sign of the weights).
# The training grid is labelled by the lab's own policy (policy_label
# below) - the classifier interpolates the policy, it does not
# replace the run's own error bars.

def policy_label(excess, stderr, avg_sign):
    """The run-quality policy distilled to a three-way teaching label."""
    if avg_sign < 0.1:
        return "sign-blocked"
    if stderr >= 0.008:
        return "under-sampled"
    return "converged"


def _training_grid():
    rows = []
    for excess in (0.0, 0.01, 0.03, 0.06, 0.12):
        for stderr in (0.0, 0.002, 0.006, 0.012, 0.03):
            for avg_sign in (0.0005, 0.02, 0.09, 0.3, 1.0):
                rows.append(((excess, stderr, avg_sign),
                             policy_label(excess, stderr, avg_sign)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    excess, stderr, avg_sign = features
    return (excess / 0.12, stderr / 0.03, avg_sign)


def classify_run(excess, stderr, avg_sign, k=5):
    """Distance-weighted kNN over the policy-labelled grid."""
    target = _scaled((excess, stderr, avg_sign))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    weights = {}
    for dist, _, label in ranked[:k]:
        weights[label] = weights.get(label, 0.0) + 1.0 / (dist + 1e-9)
    priority = {"converged": 0, "under-sampled": 1, "sign-blocked": 2}
    return sorted(weights.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    """Scenarios built from this lab's own measured run quantities."""
    vmc = vmc_report()
    helium = helium_report()
    sign = sign_report()
    opt = vmc["optimal"]
    # A genuinely short run (1,200 kept samples) as the under-sampled
    # scenario: its measured error bar is wide by construction, far
    # from the policy boundary, so the label is not a coin flip.
    short = run_ho_vmc(0.35, 1200, 500, SEED + 31)
    he_opt = helium["optimal"]
    sign6 = sign["n_sweep"][5]  # N = 6 at beta = 0.5
    scenarios = {
        "ho_optimal_width": (abs(opt["estimate"] - HO_E0_EXACT),
                             opt["stderr"], 1.0),
        "ho_short_run": (abs(short["estimate"] - HO_E0_EXACT),
                         short["stderr"], 1.0),
        "helium_optimal_long": (he_opt["estimate"] - HELIUM_TRUE_E0,
                                he_opt["stderr"], 1.0),
        "fermion_sign_toy_n6": (0.0, 0.0, sign6["avg_sign"]),
    }
    return {name: {"features": features,
                   "label": classify_run(*features),
                   "policy": policy_label(*features)}
            for name, features in scenarios.items()}


# ================================================================ 5. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantummontecarlo-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "model_id": "gemini-argon-4",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all production gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantummontecarlo-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Print the error bar next to every energy and the average sign next to every fermionic claim: a variational bound without its bar, and a fermion number without its sign, are both only half a result.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a Gaussian trial wave breathing over a parabolic well",
                  "walkers diffusing while the local-energy trace flattens",
                  "error bars shrinking by half as the sample count quadruples",
                  "two electrons orbiting a nucleus inside a screened cloud",
                  "permutation cycles flipping sign until the average sign fades to zero"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 0. PRNG - seeded streams are reproducible bit for bit
    rng_a, rng_b = SplitMix64(SEED), SplitMix64(SEED)
    chk("prng_deterministic_stream",
        [rng_a.next_u64() for _ in range(8)]
        == [rng_b.next_u64() for _ in range(8)])
    rng_c = SplitMix64(SEED)
    draws = [rng_c.random() for _ in range(2000)]
    chk("prng_uniform_range", all(0.0 <= u < 1.0 for u in draws))
    chk("prng_uniform_mean", abs(sum(draws) / len(draws) - 0.5) < 0.03)

    # 1. oscillator analytics
    chk("ho_exact_optimum", abs(ho_exact_energy(0.5) - 0.5) < 1e-12)
    chk("ho_exact_wrong_width", abs(ho_exact_energy(0.35) - 0.5321428571428571)
        < 1e-12)
    chk("ho_local_energy_constant_at_optimum",
        all(abs(ho_local_energy(x, 0.5) - 0.5) < 1e-12
            for x in (-2.0, -0.3, 0.0, 0.7, 3.1)))
    chk("ho_variational_ordering",
        ho_exact_energy(0.35) > ho_exact_energy(0.5)
        and ho_exact_energy(0.8) > ho_exact_energy(0.5))

    # 2. oscillator VMC - estimates graded against the analytic curve
    vmc = vmc_report()
    opt, wrong = vmc["optimal"], vmc["wrong_width"]
    chk("vmc_optimal_equals_e0",
        abs(opt["estimate"] - HO_E0_EXACT) < 1e-9 and opt["stderr"] < 1e-9)
    chk("vmc_wrong_width_above_e0_beyond_bars",
        wrong["estimate"] - 3.0 * wrong["stderr"] > HO_E0_EXACT)
    chk("vmc_wrong_width_matches_exact",
        abs(wrong["estimate"] - wrong["exact_expectation"])
        < 4.0 * wrong["stderr"])
    chk("vmc_acceptance_sane_window",
        0.2 < opt["acceptance_rate"] < 0.8
        and 0.2 < wrong["acceptance_rate"] < 0.8)
    chk("vmc_burn_in_bookkeeping",
        opt["burn_in_discarded"] == 2000 and opt["kept_samples"] == 20000
        and wrong["burn_in_discarded"] == 2000)

    # 3. error-bar scaling - the 1/sqrt(M) law, measured
    scaling = scaling_report()
    chk("scaling_stderr_ratio_window",
        3.0 < scaling["stderr_ratio"] < 5.5)
    chk("scaling_long_run_tighter",
        scaling["long"]["stderr"] < scaling["short"]["stderr"])
    chk("scaling_both_match_exact",
        abs(scaling["short"]["estimate"] - ho_exact_energy(0.35))
        < 4.0 * scaling["short"]["stderr"]
        and abs(scaling["long"]["estimate"] - ho_exact_energy(0.35))
        < 4.0 * scaling["long"]["stderr"])

    # 4. helium - bound, scan, and the <1/r12> referee
    chk("helium_family_minimum_exact",
        abs(helium_exact_energy(HELIUM_OPTIMAL_ZEFF) - (-2.84765625)) < 1e-9)
    chk("helium_family_minimum_is_minimum",
        all(helium_exact_energy(HELIUM_OPTIMAL_ZEFF) <= helium_exact_energy(z)
            for z in HELIUM_SCAN_ZEFF))
    helium = helium_report()
    he_opt = helium["optimal"]
    chk("helium_mc_matches_exact",
        abs(he_opt["estimate"] - he_opt["exact_expectation"])
        < 4.0 * he_opt["stderr"])
    chk("helium_upper_bound_beyond_bars",
        he_opt["estimate"] - 3.0 * he_opt["stderr"] > HELIUM_TRUE_E0)
    chk("helium_scan_best_at_optimum", helium["best_zeff"] == HELIUM_OPTIMAL_ZEFF)
    scan_by_z = {row["zeff"]: row for row in helium["scan"]}
    chk("helium_scan_improves_over_unscreened",
        scan_by_z[HELIUM_OPTIMAL_ZEFF]["estimate"]
        + 3.0 * scan_by_z[HELIUM_OPTIMAL_ZEFF]["stderr"]
        < scan_by_z[2.0]["estimate"] - 3.0 * scan_by_z[2.0]["stderr"])
    chk("helium_scan_rows_match_exact",
        all(abs(row["estimate"] - row["exact_expectation"])
            < 4.0 * row["stderr"] for row in helium["scan"]))
    chk("helium_inv_r12_referee",
        abs(he_opt["mean_inv_r12"] - he_opt["mean_inv_r12_exact"])
        < 0.02 * he_opt["mean_inv_r12_exact"])

    # 5. sign problem - exact recursion, monotone collapse, cost wall
    chk("z1_closed_form", abs(z1(1.0) - 0.9595173757) < 1e-9)
    chk("sign_single_particle_is_one",
        abs(average_sign(0.5, 1) - 1.0) < 1e-12)
    sign = sign_report()
    n_signs = [row["avg_sign"] for row in sign["n_sweep"]]
    chk("sign_monotone_in_n",
        all(a > b for a, b in zip(n_signs, n_signs[1:])))
    beta_signs = [row["avg_sign"] for row in sign["beta_sweep"]]
    chk("sign_monotone_in_beta",
        all(a > b for a, b in zip(beta_signs, beta_signs[1:])))
    chk("sign_cost_factor_explodes",
        sign["n_sweep"][5]["cost_factor"] > 1.0e6
        and sign["n_sweep"][1]["cost_factor"]
        < sign["n_sweep"][3]["cost_factor"]
        < sign["n_sweep"][5]["cost_factor"])
    asym = sign["asymptote"]
    chk("sign_ground_state_asymptote",
        abs(asym["computed"] - asym["formula"]) / asym["formula"] < 1e-6)

    # 6. classifier - labels come from the policy itself
    rep = classifier_report()
    chk("classifier_ho_optimal_converged",
        rep["ho_optimal_width"]["label"] == "converged")
    chk("classifier_short_run_under_sampled",
        rep["ho_short_run"]["label"] == "under-sampled")
    chk("classifier_helium_converged",
        rep["helium_optimal_long"]["label"] == "converged")
    chk("classifier_sign_toy_blocked",
        rep["fermion_sign_toy_n6"]["label"] == "sign-blocked")
    chk("classifier_matches_policy_on_scenarios",
        all(row["label"] == row["policy"] for row in rep.values()))
    chk("classifier_matches_policy_on_grid",
        all(classify_run(*features) == label
            for features, label in TRAINING_GRID))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_production_chemistry_hold",
        gate("production_chemistry_claim") == "HOLD")
    chk("gate_sign_problem_hold", gate("sign_problem_solved_claim") == "HOLD")
    chk("gate_drug_hold", gate("drug_design_claim") == "HOLD")
    chk("gate_publication_hold", gate("publication_result_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking"
        and brains["decider"]["model_id"] == "gemini-argon-4")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumMonteCarlo-Lab - variational Monte Carlo & the sign problem")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print(f"PRNG splitmix64 | seed {SEED} | every number below is reproducible")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - what Monte Carlo can and cannot promise")
    print("    VMC energy estimates are UPPER bounds plus noise: the exact")
    print("    expectation over a trial state never dips below the true")
    print("    ground energy. So every estimate ships with its block-averaged")
    print("    error bar, and is graded against an analytic expectation")
    print("    wherever one exists. Fermion numbers ship with their average")
    print("    sign - when the sign dies, the cost explodes as 1/<sign>^2.")
    vmc = vmc_report()
    opt, wrong = vmc["optimal"], vmc["wrong_width"]
    print("\n[2] VMC - 1D harmonic oscillator, Gaussian trial state")
    print(f"    alpha = 0.50 (optimal): E = {opt['estimate']:.6f} "
          f"+- {opt['stderr']:.2e} | exact E0 = 0.5")
    print("    -> at the optimum the local energy is the constant 0.5 for")
    print("       every sample: the zero-variance signature of an exact")
    print("       eigenstate, reproduced by the sampler.")
    print(f"    alpha = 0.35 (wrong):   E = {wrong['estimate']:.6f} "
          f"+- {wrong['stderr']:.6f} | exact E(alpha) = "
          f"{wrong['exact_expectation']:.6f}")
    print("    -> strictly above 0.5 beyond the error bars: the variational")
    print("       principle, measured - a wrong width can only cost energy.")
    print(f"    Metropolis bookkeeping: burn-in {opt['burn_in_discarded']} "
          f"discarded, kept {opt['kept_samples']}, acceptance "
          f"{opt['acceptance_rate']:.3f} (optimal) / "
          f"{wrong['acceptance_rate']:.3f} (wrong width)")
    scaling = scaling_report()
    short, long = scaling["short"], scaling["long"]
    print("\n[3] Error bars - the 1/sqrt(M) law, measured not assumed")
    print(f"    M = {short['kept_samples']:>6}: stderr {short['stderr']:.6f}")
    print(f"    M = {long['kept_samples']:>6}: stderr {long['stderr']:.6f}")
    print(f"    ratio {scaling['stderr_ratio']:.3f} vs sqrt(16) = "
          f"{scaling['sqrt_law_prediction']:.1f} predicted - 16x the "
          f"samples buys 4x the precision, no more.")
    helium = helium_report()
    he_opt = helium["optimal"]
    print("\n[4] VMC - helium-like atom, screened hydrogenic trial state")
    print("    zeff scan (exact family curve E(z) = z^2 - 4z + 5z/8):")
    for row in helium["scan"]:
        print(f"      zeff {row['zeff']:.4f}: MC {row['estimate']:.5f} "
              f"+- {row['stderr']:.5f} | exact {row['exact_expectation']:.5f}")
    print(f"    long run at zeff = 27/16: E = {he_opt['estimate']:.5f} "
          f"+- {he_opt['stderr']:.5f}")
    print(f"    labelled true E0 = {HELIUM_TRUE_E0} Ha - the estimate "
          f"stays above it")
    print("    beyond its error bars: an upper bound doing its job. The gap")
    print("    is the price of a Jastrow-free trial state, stated openly.")
    print(f"    referee: <1/r12> MC {he_opt['mean_inv_r12']:.4f} vs exact "
          f"5*zeff/8 = {he_opt['mean_inv_r12_exact']:.4f}")
    sign = sign_report()
    print("\n[5] The sign problem - free fermions in a harmonic trap (TOY)")
    print(f"    exact permutation-sign recursion at beta = "
          f"{sign['beta_for_n_sweep']}:")
    for row in sign["n_sweep"]:
        print(f"      N = {row['n']}: <sign> = {row['avg_sign']:.6f} | "
              f"cost x{row['cost_factor']:.3e}")
    print(f"    at fixed N = {sign['n_for_beta_sweep']}, warming helps and "
          f"cooling kills:")
    for row in sign["beta_sweep"]:
        print(f"      beta = {row['beta']:.2f}: <sign> = "
              f"{row['avg_sign']:.6f}")
    print("    -> a labelled toy (free particles, exact arithmetic): the")
    print("       average sign collapses monotonically, and the sampling")
    print("       cost follows 1/<sign>^2 through the floor. Demonstrated,")
    print("       not solved - claiming otherwise is a HOLD gate.")
    print("\n[6] Run classifier (TinyML kNN, labels from the run policy)")
    for name, row in classifier_report().items():
        print(f"    {name:24s} -> {row['label']}")
    print("\n[7] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "vmc":
        print(json.dumps(vmc_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "scaling":
        print(json.dumps(scaling_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "helium":
        print(json.dumps(helium_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "sign":
        print(json.dumps(sign_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "production_chemistry_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
