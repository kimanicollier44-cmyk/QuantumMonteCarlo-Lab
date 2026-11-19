"""Tests for QuantumMonteCarlo-Lab. Every test grades a computed
number against a closed form or an independent route in main.py."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as qmc


def test_prng_deterministic():
    a, b = qmc.SplitMix64(qmc.SEED), qmc.SplitMix64(qmc.SEED)
    assert [a.next_u64() for _ in range(16)] == [b.next_u64() for _ in range(16)]


def test_prng_uniform_moments():
    rng = qmc.SplitMix64(qmc.SEED)
    draws = [rng.random() for _ in range(4000)]
    assert all(0.0 <= u < 1.0 for u in draws)
    assert abs(sum(draws) / len(draws) - 0.5) < 0.02


def test_ho_exact_curve():
    assert abs(qmc.ho_exact_energy(0.5) - 0.5) < 1e-12
    assert abs(qmc.ho_exact_energy(0.35) - 0.5321428571428571) < 1e-12
    assert qmc.ho_exact_energy(0.35) > 0.5
    assert qmc.ho_exact_energy(0.8) > 0.5


def test_ho_local_energy_zero_variance_at_optimum():
    for x in (-3.0, -0.25, 0.0, 1.5):
        assert abs(qmc.ho_local_energy(x, 0.5) - 0.5) < 1e-12


def test_vmc_optimal_is_exact():
    run = qmc.vmc_report()["optimal"]
    assert abs(run["estimate"] - 0.5) < 1e-9
    assert run["stderr"] < 1e-9


def test_vmc_wrong_width_variational():
    run = qmc.vmc_report()["wrong_width"]
    assert run["estimate"] - 3.0 * run["stderr"] > 0.5
    assert abs(run["estimate"] - run["exact_expectation"]) < 4.0 * run["stderr"]


def test_vmc_bookkeeping():
    run = qmc.vmc_report()["optimal"]
    assert run["burn_in_discarded"] == 2000
    assert run["kept_samples"] == 20000
    assert 0.2 < run["acceptance_rate"] < 0.8


def test_scaling_ratio_window():
    rep = qmc.scaling_report()
    assert 3.0 < rep["stderr_ratio"] < 5.5
    assert rep["long"]["stderr"] < rep["short"]["stderr"]


def test_helium_family_curve():
    assert abs(qmc.helium_exact_energy(1.6875) - (-2.84765625)) < 1e-9
    assert abs(qmc.helium_exact_energy(2.0) - (-2.75)) < 1e-12


def test_helium_mc_matches_exact_and_bounds():
    run = qmc.helium_report()["optimal"]
    assert abs(run["estimate"] - run["exact_expectation"]) < 4.0 * run["stderr"]
    assert run["estimate"] - 3.0 * run["stderr"] > qmc.HELIUM_TRUE_E0


def test_helium_scan_finds_optimum():
    rep = qmc.helium_report()
    assert rep["best_zeff"] == 1.6875
    by_z = {row["zeff"]: row for row in rep["scan"]}
    assert (by_z[1.6875]["estimate"] + 3.0 * by_z[1.6875]["stderr"]
            < by_z[2.0]["estimate"] - 3.0 * by_z[2.0]["stderr"])


def test_helium_inv_r12_referee():
    run = qmc.helium_report()["optimal"]
    assert abs(run["mean_inv_r12"] - 0.625 * 1.6875) < 0.02 * 0.625 * 1.6875


def test_sign_recursion_values():
    assert abs(qmc.z1(1.0) - 0.9595173757) < 1e-9
    assert abs(qmc.average_sign(0.5, 1) - 1.0) < 1e-12
    assert abs(qmc.average_sign(0.5, 2) - 0.606531) < 1e-5


def test_sign_monotone_collapse():
    rep = qmc.sign_report()
    signs = [row["avg_sign"] for row in rep["n_sweep"]]
    assert all(a > b for a, b in zip(signs, signs[1:]))
    betas = [row["avg_sign"] for row in rep["beta_sweep"]]
    assert all(a > b for a, b in zip(betas, betas[1:]))
    assert rep["n_sweep"][5]["cost_factor"] > 1.0e6


def test_sign_asymptote():
    asym = qmc.sign_report()["asymptote"]
    assert abs(asym["computed"] - asym["formula"]) / asym["formula"] < 1e-6


def test_classifier_scenarios():
    rep = qmc.classifier_report()
    assert rep["ho_optimal_width"]["label"] == "converged"
    assert rep["ho_short_run"]["label"] == "under-sampled"
    assert rep["helium_optimal_long"]["label"] == "converged"
    assert rep["fermion_sign_toy_n6"]["label"] == "sign-blocked"
    assert all(row["label"] == row["policy"] for row in rep.values())


def test_classifier_grid_agreement():
    assert all(qmc.classify_run(*f) == label
               for f, label in qmc.TRAINING_GRID)


def test_gates():
    assert len(qmc.HOLD_GATES) == 14
    assert qmc.gate("production_chemistry_claim") == "HOLD"
    assert qmc.gate("sign_problem_solved_claim") == "HOLD"
    assert qmc.gate("run_simulation") == "PROCEED"


def test_seats():
    brains = qmc.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["model_id"] == "gemini-argon-4"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False and brains["flow"]["live"] == 0


def test_checks_all_pass():
    checks = qmc.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


def test_hold_demo_exit_2():
    proc = subprocess.run(
        [sys.executable, "main.py", "hold-demo"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "HOLD" in proc.stdout


def test_demo_deterministic():
    root = Path(__file__).resolve().parent.parent
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "main.py", "demo"], cwd=root,
            capture_output=True, text=True)
        assert proc.returncode == 0
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
