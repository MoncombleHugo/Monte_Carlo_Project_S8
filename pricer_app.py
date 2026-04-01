import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, qmc


@dataclass
class PriceResult:
    method: str
    price: float
    se: float
    ci_low: float
    ci_high: float
    ci_width: float
    runtime_s: float


def ci95_from_samples(x: np.ndarray) -> Tuple[float, float, float, float]:
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    low = mean - 1.96 * se
    high = mean + 1.96 * se
    return mean, se, low, high


def black_scholes_call(s0: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def make_corr_matrix(d: int, kind: str, rho: float, rho2: float) -> np.ndarray:
    if kind == "Identity":
        corr = np.eye(d)
    elif kind == "Constant":
        rho_max_neg = -1.0 / max(d - 1, 1)
        rho_clip = float(np.clip(rho, rho_max_neg + 1e-6, 0.999))
        corr = np.full((d, d), rho_clip)
        np.fill_diagonal(corr, 1.0)
    elif kind == "Toeplitz":
        base = float(np.clip(abs(rho), 0.0, 0.999))
        idx = np.arange(d)
        corr = base ** np.abs(idx[:, None] - idx[None, :])
        np.fill_diagonal(corr, 1.0)
    elif kind == "Two-Block":
        half = d // 2
        corr = np.full((d, d), rho2)
        corr[:half, :half] = rho
        corr[half:, half:] = rho
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.eye(d)

    # Ensure positive semidefinite by eigenvalue clipping.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-10, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    dvec = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(dvec, dvec)
    return corr_psd


def chol_with_jitter(a: np.ndarray) -> np.ndarray:
    jitter = 1e-12
    eye = np.eye(a.shape[0])
    for _ in range(8):
        try:
            return np.linalg.cholesky(a + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Could not compute Cholesky factor.")


def simulate_payoff(
    z: np.ndarray,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    z_corr = z @ lmat.T
    drift = (r - 0.5 * sigma_vec * sigma_vec) * t
    diffusion = sigma_vec * math.sqrt(t) * z_corr
    st_mat = s0_vec * np.exp(drift + diffusion)
    basket_t = st_mat @ w_vec
    discounted_payoff = math.exp(-r * t) * np.maximum(basket_t - k, 0.0)
    # Control variate with known expectation E[e^{-rT} B(T)] = w dot S0
    control = math.exp(-r * t) * basket_t
    return discounted_payoff, control


def price_mc(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> PriceResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(s0_vec)))
    x, _ = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
    mean, se, low, high = ci95_from_samples(x)
    return PriceResult("MC", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_mc_antithetic(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> PriceResult:
    t0 = time.perf_counter()
    n2 = n // 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n2, len(s0_vec)))
    x1, _ = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
    x2, _ = simulate_payoff(-z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
    pair = 0.5 * (x1 + x2)
    mean, se, low, high = ci95_from_samples(pair)
    return PriceResult("MC Antithetic", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_mc_control_variate(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
    antithetic: bool,
) -> PriceResult:
    t0 = time.perf_counter()
    c_mean = float(np.dot(w_vec, s0_vec))
    d = len(s0_vec)
    rng = np.random.default_rng(seed)

    if antithetic:
        n2 = n // 2
        z = rng.standard_normal((n2, d))
        x1, c1 = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        x2, c2 = simulate_payoff(-z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        x = 0.5 * (x1 + x2)
        c = 0.5 * (c1 + c2)
        method_name = "MC Anti + CV"
    else:
        z = rng.standard_normal((n, d))
        x, c = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        method_name = "MC + CV"

    cov = np.cov(x, c, ddof=1)
    beta = float(cov[0, 1] / max(cov[1, 1], 1e-15))
    x_cv = x - beta * (c - c_mean)
    mean, se, low, high = ci95_from_samples(x_cv)
    return PriceResult(method_name, mean, se, low, high, high - low, time.perf_counter() - t0)


def translation_direction(
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> np.ndarray:
    drift = (r - 0.5 * sigma_vec * sigma_vec) * t
    basket_gradient_corr = w_vec * s0_vec * np.exp(drift) * sigma_vec * math.sqrt(t)
    direction = basket_gradient_corr @ lmat
    norm_dir = float(np.linalg.norm(direction))

    if norm_dir <= 1e-15:
        return np.full(len(s0_vec), 1.0 / math.sqrt(len(s0_vec)))

    return direction / norm_dir


def select_theta_translation(
    n_pilot: int,
    seed: int,
    theta_grid: np.ndarray,
    direction: np.ndarray,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> float:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_pilot, len(s0_vec)))

    best_theta = float(theta_grid[0])
    best_var = float("inf")
    projection = z @ direction

    for theta in theta_grid:
        shift = theta * direction
        x_shifted, _ = simulate_payoff(z + shift, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        weights = np.exp(-theta * projection - 0.5 * theta * theta)
        y = x_shifted * weights
        var = float(np.var(y, ddof=1))

        if var < best_var:
            best_var = var
            best_theta = float(theta)

    return best_theta


def price_mc_translation(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
    n_pilot: int | None = None,
    theta_grid: np.ndarray | None = None,
) -> PriceResult:
    t0 = time.perf_counter()
    d = len(s0_vec)

    if n_pilot is None:
        n_pilot = 50_000 if d == 1 else 20_000

    if theta_grid is None:
        theta_grid = np.linspace(-2.0, 2.0, 101 if d == 1 else 41)

    direction = translation_direction(s0_vec, sigma_vec, w_vec, r, t, lmat)
    theta = select_theta_translation(
        n_pilot=n_pilot,
        seed=seed + 1,
        theta_grid=theta_grid,
        direction=direction,
        s0_vec=s0_vec,
        sigma_vec=sigma_vec,
        w_vec=w_vec,
        k=k,
        r=r,
        t=t,
        lmat=lmat,
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, d))
    shift = theta * direction
    x_shifted, _ = simulate_payoff(z + shift, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
    weights = np.exp(-theta * (z @ direction) - 0.5 * theta * theta)
    y = x_shifted * weights

    mean, se, low, high = ci95_from_samples(y)
    return PriceResult("MC Translation", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_rqmc_icdf(
    n: int,
    r_repl: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> PriceResult:
    t0 = time.perf_counter()
    d = len(s0_vec)
    m = int(math.ceil(math.log2(max(n, 2))))
    estimates = np.empty(r_repl)

    for rep in range(r_repl):
        eng = qmc.Sobol(d=d, scramble=True, seed=seed + rep)
        u = eng.random_base2(m=m)[:n, :]
        u = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        z = norm.ppf(u)
        x, _ = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        estimates[rep] = float(np.mean(x))

    mean, se, low, high = ci95_from_samples(estimates)
    return PriceResult("RQMC ICDF", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_rqmc_truncated_weighted(
    n: int,
    r_repl: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
    a: float,
) -> PriceResult:
    t0 = time.perf_counter()
    d = len(s0_vec)
    m = int(math.ceil(math.log2(max(n, 2))))
    estimates = np.empty(r_repl)

    for rep in range(r_repl):
        eng = qmc.Sobol(d=d, scramble=True, seed=seed + rep)
        u = eng.random_base2(m=m)[:n, :]
        u = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        z = -a + 2.0 * a * u
        x, _ = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        weights = np.prod((2.0 * a) * norm.pdf(z), axis=1)
        y = x * weights
        estimates[rep] = float(np.mean(y))

    mean, se, low, high = ci95_from_samples(estimates)
    return PriceResult("RQMC Truncated", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_mc_stratified_naive(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    lmat: np.ndarray,
) -> PriceResult:
    t0 = time.perf_counter()
    d = len(s0_vec)
    rng = np.random.default_rng(seed)
    n_batches = 50
    steps = n // n_batches
    batch_means = []

    for _ in range(n_batches):
        u_base = rng.uniform(0.0, 1.0, steps)
        u_strat = (np.arange(steps) + u_base) / steps
        z1 = norm.ppf(u_strat)
        z_rest = rng.standard_normal((steps, d - 1))
        z = np.column_stack([z1, z_rest])
        payoffs, _ = simulate_payoff(z, s0_vec, sigma_vec, w_vec, k, r, t, lmat)
        batch_means.append(np.mean(payoffs))

    mean, se, low, high = ci95_from_samples(np.asarray(batch_means))
    return PriceResult("Stratif. Naive (Z1)", mean, se, low, high, high - low, time.perf_counter() - t0)


def price_mc_stratified_pca(
    n: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    corr: np.ndarray,
) -> PriceResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    # Build sqrt(corr) in PCA basis: z_corr = y @ A.T with y ~ N(0, I_d)
    # and A = Q diag(sqrt(lambda)). Stratification is applied on y[:, 0].
    a_mat = eigvecs @ np.diag(np.sqrt(eigvals))

    n_batches = 50
    steps = n // n_batches
    batch_means = []

    for _ in range(n_batches):
        u_strat = (np.arange(steps) + rng.uniform(0.0, 1.0, steps)) / steps
        z_strat = norm.ppf(np.clip(u_strat, np.finfo(float).eps, 1.0 - np.finfo(float).eps))

        y = rng.standard_normal((steps, len(s0_vec)))
        y[:, 0] = z_strat
        z_final = y @ a_mat.T

        drift = (r - 0.5 * sigma_vec * sigma_vec) * t
        diffusion = sigma_vec * math.sqrt(t) * z_final
        st_mat = s0_vec * np.exp(drift + diffusion)
        payoffs = math.exp(-r * t) * np.maximum(st_mat @ w_vec - k, 0.0)
        batch_means.append(np.mean(payoffs))

    mean, se, low, high = ci95_from_samples(np.asarray(batch_means))
    return PriceResult("Stratif. PCA", mean, se, low, high, high - low, time.perf_counter() - t0)


def run_methods(
    methods: List[str],
    n_mc: int,
    n_rqmc: int,
    r_repl: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    corr: np.ndarray,
    trunc_a: float,
    include_bs: bool,
) -> List[PriceResult]:
    lmat = chol_with_jitter(corr)
    results: List[PriceResult] = []

    if include_bs and len(s0_vec) == 1:
        t0 = time.perf_counter()
        exact = black_scholes_call(float(s0_vec[0]), k, t, r, float(sigma_vec[0]))
        results.append(PriceResult("Black-Scholes", exact, 0.0, exact, exact, 0.0, time.perf_counter() - t0))

    for m in methods:
        if m == "MC":
            results.append(price_mc(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat))
        elif m == "MC Antithetic":
            results.append(price_mc_antithetic(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat))
        elif m == "MC Translation":
            results.append(price_mc_translation(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat))
        elif m == "MC + CV":
            results.append(price_mc_control_variate(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat, antithetic=False))
        elif m == "MC Anti + CV":
            results.append(price_mc_control_variate(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat, antithetic=True))
        elif m == "RQMC ICDF":
            results.append(price_rqmc_icdf(n_rqmc, r_repl, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat))
        elif m == "RQMC Truncated":
            results.append(
                price_rqmc_truncated_weighted(
                    n_rqmc,
                    r_repl,
                    seed,
                    s0_vec,
                    sigma_vec,
                    w_vec,
                    k,
                    r,
                    t,
                    lmat,
                    trunc_a,
                )
            )
        elif m == "Stratif. Naive (Z1)":
            results.append(price_mc_stratified_naive(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, lmat))
        elif m == "Stratif. PCA":
            results.append(price_mc_stratified_pca(n_mc, seed, s0_vec, sigma_vec, w_vec, k, r, t, corr))

    return results


def select_best_method_auto(
    candidate_methods: List[str],
    n_mc: int,
    n_rqmc: int,
    r_repl: int,
    seed: int,
    s0_vec: np.ndarray,
    sigma_vec: np.ndarray,
    w_vec: np.ndarray,
    k: float,
    r: float,
    t: float,
    corr: np.ndarray,
    trunc_a: float,
) -> Tuple[str, pd.DataFrame]:
    del candidate_methods, seed, r, t, trunc_a

    d = len(s0_vec)
    basket_spot = float(np.dot(w_vec, s0_vec))
    moneyness = basket_spot / max(float(k), 1e-12)
    log_moneyness = float(math.log(max(moneyness, 1e-12)))
    sigma_eff = float(np.dot(w_vec, sigma_vec))

    off_diag = corr[~np.eye(d, dtype=bool)] if d > 1 else np.array([0.0])
    mean_abs_corr = float(np.mean(np.abs(off_diag)))

    rqmc_budget = int(n_rqmc) * int(r_repl)
    high_rqmc_budget = rqmc_budget >= 90_000
    rich_mc_budget = int(n_mc) >= 120_000

    decision_rules: List[str] = []

    if d == 1:
        if moneyness <= 0.9 or moneyness >= 1.15:
            best_method = "MC Translation"
            decision_rules.append("1D loin de la monnaie: translation pour mieux capter les événements rares")
        elif high_rqmc_budget and 0.10 <= sigma_eff <= 0.45:
            best_method = "RQMC ICDF"
            decision_rules.append("1D quasi ATM avec budget RQMC suffisant: RQMC ICDF")
        else:
            best_method = "MC Anti + CV"
            decision_rules.append("1D régime standard: MC antithétique + variable de contrôle")
    elif d <= 8:
        if mean_abs_corr >= 0.55 and rich_mc_budget:
            best_method = "Stratif. PCA"
            decision_rules.append("corrélation forte en faible dimension: stratification PCA")
        elif high_rqmc_budget and sigma_eff <= 0.40:
            best_method = "RQMC ICDF"
            decision_rules.append("faible dimension avec budget RQMC élevé: RQMC ICDF")
        else:
            best_method = "MC Anti + CV"
            decision_rules.append("faible dimension sans signal dominant: MC Anti + CV")
    elif d <= 20:
        if high_rqmc_budget and mean_abs_corr <= 0.60 and abs(log_moneyness) <= 0.20:
            best_method = "RQMC ICDF"
            decision_rules.append("dimension intermédiaire et structure régulière: RQMC ICDF")
        elif mean_abs_corr >= 0.65 and rich_mc_budget:
            best_method = "Stratif. PCA"
            decision_rules.append("corrélation marquée: stratification PCA")
        else:
            best_method = "MC Anti + CV"
            decision_rules.append("dimension intermédiaire robuste: MC Anti + CV")
    else:
        if high_rqmc_budget and d <= 40 and mean_abs_corr <= 0.45 and sigma_eff <= 0.30:
            best_method = "RQMC ICDF"
            decision_rules.append("haute dimension modérée avec faible corrélation: RQMC ICDF")
        else:
            best_method = "MC Anti + CV"
            decision_rules.append("haute dimension: méthode la plus robuste en temps et stabilité")

    if best_method not in {
        "MC",
        "MC Translation",
        "MC Anti + CV",
        "RQMC ICDF",
        "RQMC Truncated",
        "Stratif. PCA",
    }:
        best_method = "MC Anti + CV"
        decision_rules.append("fallback de sécurité")

    diag = pd.DataFrame(
        [
            {"Metric": "Dimension", "Value": d},
            {"Metric": "Moneyness (S0/K)", "Value": round(moneyness, 6)},
            {"Metric": "log-moneyness", "Value": round(log_moneyness, 6)},
            {"Metric": "Volatilité efficace", "Value": round(sigma_eff, 6)},
            {"Metric": "Moyenne |corr| hors diagonale", "Value": round(mean_abs_corr, 6)},
            {"Metric": "Budget MC", "Value": int(n_mc)},
            {"Metric": "Budget RQMC total (N x R)", "Value": int(rqmc_budget)},
            {"Metric": "Méthode choisie", "Value": best_method},
            {"Metric": "Règle appliquée", "Value": " | ".join(decision_rules)},
        ]
    )

    return best_method, diag


def results_to_frame(results: List[PriceResult], ref_price: float | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Method": r.method,
            "Price": r.price,
            "SE": r.se,
            "CI95 Low": r.ci_low,
            "CI95 High": r.ci_high,
            "CI95 Width": r.ci_width,
            "Runtime (s)": r.runtime_s,
        }
        if ref_price is not None:
            row["Abs Diff vs Ref"] = abs(r.price - ref_price)
        rows.append(row)
    return pd.DataFrame(rows)


def run_sensitivity(
    param_name: str,
    param_grid: np.ndarray,
    selected_method: str,
    common_inputs: Dict,
) -> pd.DataFrame:
    rows = []

    for val in param_grid:
        local = dict(common_inputs)
        local[param_name] = float(val)

        d = int(local["d"])
        s0_vec = np.full(d, local["s0"])
        sigma_vec = np.full(d, local["sigma"])
        w_vec = np.full(d, 1.0 / d)

        corr = make_corr_matrix(d, local["corr_kind"], local["corr_rho"], local["corr_rho2"])
        out = run_methods(
            methods=[selected_method],
            n_mc=local["n_mc_sens"],
            n_rqmc=local["n_rqmc_sens"],
            r_repl=local["r_repl_sens"],
            seed=local["seed"],
            s0_vec=s0_vec,
            sigma_vec=sigma_vec,
            w_vec=w_vec,
            k=local["k"],
            r=local["r"],
            t=local["t"],
            corr=corr,
            trunc_a=local["trunc_a"],
            include_bs=False,
        )[0]

        rows.append(
            {
                "Param": param_name,
                "Value": float(val),
                "Method": selected_method,
                "Price": out.price,
                "SE": out.se,
                "CI95 Width": out.ci_width,
                "CI95 Low": out.ci_low,
                "CI95 High": out.ci_high,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Monte Carlo Option Pricer", layout="wide")
    st.title("Monte Carlo Option Pricer")
    st.caption("Comparaison de méthodes + étude de sensibilité (format pricer interactif)")

    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Produit", ["European 1D", "Basket Arithmetique"])

        s0 = st.number_input("Spot S0", min_value=1.0, value=100.0, step=1.0)
        k = st.number_input("Strike K", min_value=1.0, value=100.0, step=1.0)
        t = st.number_input("Maturite T", min_value=0.05, value=1.0, step=0.05)
        r = st.number_input("Taux sans risque r", value=0.10, step=0.01, format="%.4f")
        sigma = st.number_input("Volatilite sigma", min_value=0.01, value=0.20, step=0.01, format="%.4f")

        if mode == "European 1D":
            d = 1
        else:
            d = st.slider("Dimension du panier (assets)", min_value=2, max_value=60, value=10, step=1)

        st.subheader("Covariance / Correlation")
        corr_kind = st.selectbox("Type de matrice", ["Identity", "Constant", "Toeplitz", "Two-Block"])
        corr_rho = st.slider("Parametre rho", min_value=-0.9, max_value=0.95, value=0.3, step=0.05)
        corr_rho2 = st.slider("Parametre secondaire rho2", min_value=-0.5, max_value=0.8, value=0.05, step=0.05)

        st.subheader("Budgets de simulation")
        n_mc = st.number_input("N MC", min_value=1000, max_value=2_000_000, value=200_000, step=1000)
        n_rqmc = st.number_input("N RQMC", min_value=1024, max_value=262_144, value=16_384, step=1024)
        r_repl = st.slider("Repetitions RQMC", min_value=4, max_value=40, value=12, step=1)
        trunc_a = st.slider("Borne troncature a (RQMC Truncated)", min_value=2.0, max_value=10.0, value=6.0, step=0.5)
        seed = st.number_input("Seed", min_value=1, max_value=999999, value=2026, step=1)

        st.subheader("Methodes")
        run_mode = st.radio(
            "Mode de pricing",
            ["Manuel", "Auto (choix optimal)"],
            index=1,
        )
        all_methods = [
            "MC",
            "MC Antithetic",
            "MC Translation",
            "MC + CV",
            "MC Anti + CV",
            "RQMC ICDF",
            "RQMC Truncated",
            "Stratif. Naive (Z1)",
            "Stratif. PCA",
        ]
        default_methods = ["MC", "MC Translation", "MC Anti + CV", "RQMC ICDF", "RQMC Truncated"]
        methods = st.multiselect("Selection", all_methods, default=default_methods)

        run_btn = st.button("Lancer le pricing", type="primary")

    tabs = st.tabs(["Pricing", "Sensibilite"])

    corr = make_corr_matrix(d, corr_kind, corr_rho, corr_rho2)

    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Matrice de correlation")
            fig_corr = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
            fig_corr.update_layout(height=320, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig_corr, use_container_width=True)

        if run_btn:
            if run_mode == "Manuel" and not methods:
                st.warning("Selectionner au moins une methode.")
            else:
                s0_vec = np.full(d, s0)
                sigma_vec = np.full(d, sigma)
                w_vec = np.full(d, 1.0 / d)

                include_bs = d == 1 and mode == "European 1D"

                selected_methods = methods
                auto_diag = None
                if run_mode == "Auto (choix optimal)":
                    candidate_methods = [
                        "MC",
                        "MC Translation",
                        "MC Anti + CV",
                        "RQMC ICDF",
                        "RQMC Truncated",
                        "Stratif. PCA",
                    ]
                    best_method, auto_diag = select_best_method_auto(
                        candidate_methods=candidate_methods,
                        n_mc=int(n_mc),
                        n_rqmc=int(n_rqmc),
                        r_repl=int(r_repl),
                        seed=int(seed),
                        s0_vec=s0_vec,
                        sigma_vec=sigma_vec,
                        w_vec=w_vec,
                        k=k,
                        r=r,
                        t=t,
                        corr=corr,
                        trunc_a=float(trunc_a),
                    )
                    selected_methods = [best_method]
                    st.success(f"Méthode choisie automatiquement : {best_method}")

                results = run_methods(
                    methods=selected_methods,
                    n_mc=int(n_mc),
                    n_rqmc=int(n_rqmc),
                    r_repl=int(r_repl),
                    seed=int(seed),
                    s0_vec=s0_vec,
                    sigma_vec=sigma_vec,
                    w_vec=w_vec,
                    k=k,
                    r=r,
                    t=t,
                    corr=corr,
                    trunc_a=float(trunc_a),
                    include_bs=include_bs,
                )

                ref_price = None
                for r_item in results:
                    if r_item.method == "Black-Scholes":
                        ref_price = r_item.price
                        break

                if ref_price is None and "MC Anti + CV" in [x.method for x in results]:
                    ref_price = next(x.price for x in results if x.method == "MC Anti + CV")

                df = results_to_frame(results, ref_price=ref_price)

                if ref_price is not None:
                    st.metric("Prix de reference", f"{ref_price:.6f}")

                st.subheader("Tableau comparatif")
                st.dataframe(df, use_container_width=True, hide_index=True)

                if auto_diag is not None:
                    st.subheader("Diagnostic de sélection automatique (règles rapides)")
                    st.dataframe(auto_diag, use_container_width=True, hide_index=True)

                chart_df = df[df["Method"] != "Black-Scholes"].copy()
                if not chart_df.empty:
                    fig_bar = go.Figure()
                    fig_bar.add_bar(x=chart_df["Method"], y=chart_df["Price"], name="Prix")
                    fig_bar.update_layout(title="Prix par methode", yaxis_title="Prix", xaxis_title="Methode")
                    st.plotly_chart(fig_bar, use_container_width=True)

                    fig_ci = go.Figure()
                    fig_ci.add_bar(x=chart_df["Method"], y=chart_df["CI95 Width"], name="Largeur IC95")
                    fig_ci.update_layout(title="Precision statistique (largeur IC95)", yaxis_title="IC95 Width")
                    st.plotly_chart(fig_ci, use_container_width=True)

                    if "Abs Diff vs Ref" in chart_df.columns:
                        fig_err = go.Figure()
                        fig_err.add_bar(x=chart_df["Method"], y=chart_df["Abs Diff vs Ref"], name="Ecart absolu")
                        fig_err.update_layout(title="Ecart absolu a la reference", yaxis_title="|Prix - Ref|")
                        st.plotly_chart(fig_err, use_container_width=True)
        else:
            with c2:
                st.info("Configure les parametres puis clique sur 'Lancer le pricing'.")

    with tabs[1]:
        st.subheader("Etude de sensibilite")
        sens_method = st.selectbox(
            "Methode pour la sensibilite",
            ["MC", "MC Translation", "MC Anti + CV", "RQMC ICDF", "RQMC Truncated", "Stratif. Naive (Z1)", "Stratif. PCA"],
        )
        sens_param = st.selectbox("Parametre a faire varier", ["s0", "k", "sigma", "r", "t", "d", "corr_rho"])

        points = st.slider("Nombre de points", min_value=5, max_value=30, value=15)

        if sens_param == "s0":
            grid = np.linspace(max(5.0, 0.5 * s0), 1.5 * s0, points)
        elif sens_param == "k":
            grid = np.linspace(max(5.0, 0.5 * k), 1.5 * k, points)
        elif sens_param == "sigma":
            grid = np.linspace(0.05, 0.8, points)
        elif sens_param == "r":
            grid = np.linspace(-0.02, 0.30, points)
        elif sens_param == "t":
            grid = np.linspace(0.1, 5.0, points)
        elif sens_param == "d":
            grid = np.unique(np.round(np.linspace(2 if mode != "European 1D" else 1, 60, points)).astype(int)).astype(float)
        else:
            grid = np.linspace(-0.2, 0.9, points)

        n_mc_sens = st.number_input("N MC sensibilite", min_value=2000, max_value=500_000, value=80_000, step=1000)
        n_rqmc_sens = st.number_input("N RQMC sensibilite", min_value=1024, max_value=131_072, value=8192, step=1024)
        r_repl_sens = st.slider("Repetitions RQMC sensibilite", min_value=4, max_value=30, value=8)

        if st.button("Lancer la sensibilite"):
            common_inputs = {
                "s0": s0,
                "k": k,
                "sigma": sigma,
                "r": r,
                "t": t,
                "d": d,
                "corr_kind": corr_kind,
                "corr_rho": corr_rho,
                "corr_rho2": corr_rho2,
                "n_mc_sens": int(n_mc_sens),
                "n_rqmc_sens": int(n_rqmc_sens),
                "r_repl_sens": int(r_repl_sens),
                "seed": int(seed),
                "trunc_a": float(trunc_a),
            }

            df_sens = run_sensitivity(sens_param, grid, sens_method, common_inputs)
            st.dataframe(df_sens, use_container_width=True, hide_index=True)

            fig_price = px.line(df_sens, x="Value", y="Price", markers=True, title=f"Prix vs {sens_param}")
            st.plotly_chart(fig_price, use_container_width=True)

            fig_ci = px.line(df_sens, x="Value", y="CI95 Width", markers=True, title=f"Largeur IC95 vs {sens_param}")
            st.plotly_chart(fig_ci, use_container_width=True)

            st.caption("Astuce: pour les dimensions elevees, comparer RQMC ICDF et RQMC Truncated met en evidence la stabilite de la version ICDF.")

