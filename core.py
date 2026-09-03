"""
core.py — shared engine for the PortaRisk analytics suite
=========================================================
Data loading, risk models (VaR / ES + upside VaG / EG), backtests, and
Markowitz portfolio optimisation. Imported by the Home page and both tool pages.

Built on the original Portfolio Risk Engine
(github.com/Paul-Haddad/portfolio-risk-engine). Author: Paul Haddad.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import yfinance as yf
import streamlit as st

# --- Shared, colour-blind-safe palette --------------------------------------
C_DOWN = "#E15759"   # loss / downside (warm red)
C_UP = "#59A14F"     # gain / upside   (green)
C_BASE = "#4E79A7"   # neutral series  (blue)
C_ALT = "#B07AA1"    # secondary       (purple)
C_ACCENT = "#4C9BE0" # brand accent
C_GREY = "#8899AA"
RNG = np.random.default_rng(42)

TRADING_DAYS = 252


# ============================================================================
# DATA
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple, start, end) -> pd.DataFrame:
    """Adjusted-close prices, cleaned and aligned. Cached for 1 hour."""
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("No data returned — check the tickers and date range.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame()
    return prices.dropna(how="all").ffill().dropna()


def portfolio_series(prices: pd.DataFrame, weights: np.ndarray):
    """Return (asset daily returns, portfolio daily returns)."""
    asset_returns = prices.pct_change().dropna()
    port = asset_returns @ weights
    port.name = "portfolio"
    return asset_returns, port


def annualised_stats(port_returns: np.ndarray):
    r = np.asarray(port_returns)
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    return ann_ret, ann_vol, sharpe


def max_drawdown(port_returns: pd.Series):
    curve = (1 + port_returns).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    return curve, dd, dd.min()


# ============================================================================
# RISK MODELS  (downside VaR/ES + upside VaG/EG)
# ============================================================================
def historical_tails(returns, alpha):
    r = np.asarray(returns); L = -r
    dv = np.quantile(L, alpha); de = L[L >= dv].mean()
    uv = np.quantile(r, alpha); ue = r[r >= uv].mean()
    return dv, de, uv, ue


def parametric_tails(returns, alpha, dist="normal"):
    r = np.asarray(returns); mu, s = np.mean(r), np.std(r, ddof=1)
    if dist == "normal":
        z = stats.norm.ppf(alpha)
        return (-(mu - s*z), -(mu - s*stats.norm.pdf(z)/(1-alpha)),
                mu + s*z, mu + s*stats.norm.pdf(z)/(1-alpha))
    nu, loc, sc = stats.t.fit(r)
    tq = stats.t.ppf(alpha, nu)
    esf = (nu + tq**2) / (nu - 1) * stats.t.pdf(tq, nu) / (1-alpha)
    return -(loc - sc*tq), -(loc - sc*esf), loc + sc*tq, loc + sc*esf


def monte_carlo_tails(returns, alpha, n_sims=50000):
    r = np.asarray(returns); mu, s = np.mean(r), np.std(r, ddof=1)
    sim = RNG.normal(mu, s, int(n_sims)); L = -sim
    dv = np.quantile(L, alpha); de = L[L >= dv].mean()
    uv = np.quantile(sim, alpha); ue = sim[sim >= uv].mean()
    return dv, de, uv, ue


def tails_by_method(returns, alpha, mc_sims=50000) -> pd.DataFrame:
    rows = {}
    rows["Historical"] = historical_tails(returns, alpha)
    rows["Parametric-N"] = parametric_tails(returns, alpha, "normal")
    rows["Parametric-t"] = parametric_tails(returns, alpha, "student_t")
    rows["MonteCarlo"] = monte_carlo_tails(returns, alpha, mc_sims)
    df = pd.DataFrame(rows, index=["VaR", "ES", "VaG", "EG"]).T
    return df  # fractions of portfolio value


def risk_contribution(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Each asset's share of total portfolio variance (Euler/marginal decomposition)."""
    Sigma = asset_returns.cov().values
    w = np.asarray(weights)
    port_var = w @ Sigma @ w
    if port_var <= 0:
        return pd.Series(np.zeros(len(w)), index=asset_returns.columns)
    mrc = Sigma @ w                      # marginal contribution
    rc = w * mrc / port_var              # percentage contribution (sums to 1)
    return pd.Series(rc, index=asset_returns.columns)


# ============================================================================
# BACKTESTING
# ============================================================================
def rolling_var_backtest(returns, alpha, window, method="historical"):
    returns = np.asarray(returns); n = len(returns)
    vf, rz, ex = [], [], []
    for t in range(window, n):
        train = returns[t-window:t]
        var = (historical_tails(train, alpha)[0] if method == "historical"
               else parametric_tails(train, alpha, "normal")[0])
        vf.append(var); rz.append(returns[t]); ex.append(int(-returns[t] > var))
    return np.array(vf), np.array(rz), np.array(ex)


def kupiec_pof(exceptions, alpha):
    n = len(exceptions); x = int(exceptions.sum()); p = 1 - alpha
    pi = x / n if n else 0
    if x == 0 or x == n:
        return np.nan, np.nan, x, n * p
    lr = -2 * ((n-x)*np.log(1-p) + x*np.log(p) - (n-x)*np.log(1-pi) - x*np.log(pi))
    return lr, 1 - stats.chi2.cdf(lr, df=1), x, n * p


def christoffersen_independence(exceptions):
    e = np.asarray(exceptions)
    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0)); n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0)); n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))
    if (n00+n01) == 0 or (n10+n11) == 0 or (n01+n11) == 0:
        return np.nan, np.nan
    pi01 = n01/(n00+n01); pi11 = n11/(n10+n11); pi = (n01+n11)/(n00+n01+n10+n11)
    num = (1-pi)**(n00+n10) * pi**(n01+n11)
    den = (1-pi01)**n00 * pi01**n01 * (1-pi11)**n10 * pi11**n11
    lr = -2*np.log(num/den) if den > 0 else np.nan
    return lr, 1 - stats.chi2.cdf(lr, df=1)


# ============================================================================
# OPTIMISATION  (Markowitz mean-variance, long-only)
# ============================================================================
def annualised_moments(asset_returns: pd.DataFrame):
    mu = asset_returns.mean().values * TRADING_DAYS
    Sigma = asset_returns.cov().values * TRADING_DAYS
    return mu, Sigma


def portfolio_perf(w, mu, Sigma, rf=0.0):
    w = np.asarray(w)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - rf) / vol if vol else np.nan
    return ret, vol, sharpe


def _bounds_cons(n):
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    return bounds, cons


def max_sharpe(mu, Sigma, rf=0.0):
    n = len(mu); bounds, cons = _bounds_cons(n)
    neg_sharpe = lambda w: -portfolio_perf(w, mu, Sigma, rf)[2]
    res = minimize(neg_sharpe, np.repeat(1/n, n), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x


def min_variance(mu, Sigma):
    n = len(mu); bounds, cons = _bounds_cons(n)
    var = lambda w: w @ Sigma @ w
    res = minimize(var, np.repeat(1/n, n), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x


def efficient_frontier(mu, Sigma, n_points=40):
    n = len(mu); bounds, _ = _bounds_cons(n)
    lo, hi = mu.min(), mu.max()
    targets = np.linspace(lo, hi, n_points)
    vols, rets = [], []
    for tgt in targets:
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t})
        res = minimize(lambda w: w @ Sigma @ w, np.repeat(1/n, n),
                       method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            vols.append(float(np.sqrt(res.x @ Sigma @ res.x)))
            rets.append(float(res.x @ mu))
    return np.array(vols), np.array(rets)


def random_portfolios(mu, Sigma, n_sims=4000, rf=0.0):
    n = len(mu)
    W = RNG.dirichlet(np.ones(n), size=n_sims)   # random long-only weights summing to 1
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    sharpes = (rets - rf) / np.where(vols == 0, np.nan, vols)
    return vols, rets, sharpes, W
