"""
Portfolio Risk Engine — Interactive Web App
============================================
Multi-method Value-at-Risk (VaR) & Expected Shortfall (ES) with an added
upside / best-case view, live market data, and regulatory backtesting.

Built on the original Portfolio Risk Engine (github.com/Paul-Haddad/portfolio-risk-engine).
Author: Paul Haddad

Run locally:   pip install -r requirements.txt  &&  streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> pick this repo + app.py
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained, colour-blind-safe palette (extends the original engine's colours)
C_DOWN = "#C44E52"   # loss / downside  (warm red)
C_UP = "#55A868"     # gain / upside    (green)
C_BASE = "#4C72B0"   # neutral series   (blue)
C_ALT = "#8172B3"    # secondary        (purple)
C_GREY = "#8899AA"
RNG = np.random.default_rng(42)  # reproducible Monte Carlo

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .caption-small {color:#8899AA; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CORE MODELS  (identical logic to the original engine, with upside mirrors)
# ============================================================================
# In return-space, every figure below is reported as a positive magnitude:
#   down_var / down_es  -> a LOSS you could suffer (worst-case tail)
#   up_vag  / up_eg     -> a GAIN you could enjoy  (best-case tail)
# VaG = "Value-at-Gain" (upside mirror of VaR); EG = "Expected Gain" (mirror of ES).


def historical_tails(returns, alpha):
    """Historical Simulation: empirical quantiles of both tails."""
    r = np.asarray(returns)
    losses = -r
    down_var = np.quantile(losses, alpha)
    down_es = losses[losses >= down_var].mean()
    gains = r
    up_vag = np.quantile(gains, alpha)
    up_eg = gains[gains >= up_vag].mean()
    return down_var, down_es, up_vag, up_eg


def parametric_tails(returns, alpha, dist="normal"):
    """Parametric (variance-covariance). Gaussian or fat-tailed Student-t."""
    r = np.asarray(returns)
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    if dist == "normal":
        z = stats.norm.ppf(alpha)
        down_var = -(mu - sigma * z)
        down_es = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
        up_vag = mu + sigma * z
        up_eg = mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    elif dist == "student_t":
        nu, loc, scale = stats.t.fit(r)
        t_q = stats.t.ppf(alpha, nu)
        es_factor = (nu + t_q ** 2) / (nu - 1) * stats.t.pdf(t_q, nu) / (1 - alpha)
        down_var = -(loc - scale * t_q)
        down_es = -(loc - scale * es_factor)
        up_vag = loc + scale * t_q
        up_eg = loc + scale * es_factor
    else:
        raise ValueError("dist must be 'normal' or 'student_t'")
    return down_var, down_es, up_vag, up_eg


def monte_carlo_tails(returns, alpha, n_sims):
    """Monte Carlo: simulate from a fitted Gaussian, read off both tails."""
    r = np.asarray(returns)
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    sims = RNG.normal(mu, sigma, int(n_sims))
    losses = -sims
    down_var = np.quantile(losses, alpha)
    down_es = losses[losses >= down_var].mean()
    up_vag = np.quantile(sims, alpha)
    up_eg = sims[sims >= up_vag].mean()
    return down_var, down_es, up_vag, up_eg


METHODS = {
    "Historical": lambda r, a: historical_tails(r, a),
    "Parametric-N": lambda r, a: parametric_tails(r, a, "normal"),
    "Parametric-t": lambda r, a: parametric_tails(r, a, "student_t"),
    "MonteCarlo": lambda r, a: monte_carlo_tails(r, a, st.session_state.get("mc_sims", 50000)),
}


# ============================================================================
# BACKTESTING  (Kupiec POF + Christoffersen independence — unchanged logic)
# ============================================================================
def rolling_var_backtest(returns, alpha, window, method="historical"):
    returns = np.asarray(returns)
    n = len(returns)
    var_forecasts, realized, exceptions = [], [], []
    for t in range(window, n):
        train = returns[t - window:t]
        if method == "historical":
            var, *_ = historical_tails(train, alpha)
        else:
            var, *_ = parametric_tails(train, alpha, "normal")
        var_forecasts.append(var)
        realized.append(returns[t])
        exceptions.append(int(-returns[t] > var))
    return np.array(var_forecasts), np.array(realized), np.array(exceptions)


def kupiec_pof(exceptions, alpha):
    n = len(exceptions); x = int(exceptions.sum()); p = 1 - alpha
    pi = x / n if n else 0
    if x == 0 or x == n:
        return np.nan, np.nan, x, n * p
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
               - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return lr, 1 - stats.chi2.cdf(lr, df=1), x, n * p


def christoffersen_independence(exceptions):
    e = np.asarray(exceptions)
    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0)); n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0)); n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return np.nan, np.nan
    pi01 = n01 / (n00 + n01); pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    num = (1 - pi) ** (n00 + n10) * pi ** (n01 + n11)
    den = (1 - pi01) ** n00 * pi01 ** n01 * (1 - pi11) ** n10 * pi11 ** n11
    lr = -2 * np.log(num / den) if den > 0 else np.nan
    return lr, 1 - stats.chi2.cdf(lr, df=1)


# ============================================================================
# DATA PIPELINE
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("No data returned — check the tickers and date range.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame()
    prices = prices.dropna(how="all").ffill().dropna()
    return prices


# ============================================================================
# SIDEBAR — PORTFOLIO CONFIGURATION
# ============================================================================
st.sidebar.title("⚙️ Portfolio setup")

tickers_raw = st.sidebar.text_input(
    "Tickers (comma-separated)", value="SPY, TLT, GLD, QQQ, EEM",
    help="Yahoo Finance symbols, e.g. AAPL, MSFT, SPY",
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

weight_mode = st.sidebar.radio("Weights", ["Equal weight", "Custom"], horizontal=True)
if weight_mode == "Custom" and tickers:
    st.sidebar.caption("Enter weights (they'll be normalised to sum to 100%).")
    raw_w = [st.sidebar.number_input(f"{t}", min_value=0.0, value=round(1 / len(tickers), 2),
                                     step=0.05, key=f"w_{t}") for t in tickers]
    w = np.array(raw_w, dtype=float)
    weights = w / w.sum() if w.sum() > 0 else np.repeat(1 / len(tickers), len(tickers))
else:
    weights = np.repeat(1 / len(tickers), len(tickers)) if tickers else np.array([])

col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", value=pd.to_datetime("2015-01-01"))
end_date = col_b.date_input("End", value=pd.to_datetime("today"))

notional = st.sidebar.number_input("Portfolio value (USD)", min_value=1000,
                                   value=1_000_000, step=50_000, format="%d")

conf_pct = st.sidebar.select_slider(
    "Primary confidence level", options=[90, 95, 97.5, 99, 99.5], value=99,
    help="99% is the classic Basel level; 97.5% ES is the FRTB regulatory measure.",
)
alpha = conf_pct / 100

with st.sidebar.expander("Advanced settings"):
    st.session_state["mc_sims"] = st.number_input("Monte Carlo simulations", 1000, 200000,
                                                   50000, step=5000)
    bt_window = st.number_input("Backtest window (days)", 100, 1000, 500, step=50)
    bt_method = st.selectbox("Backtest method", ["historical", "parametric"])

run = st.sidebar.button("▶  Run risk analysis", type="primary", use_container_width=True)
st.sidebar.markdown(
    "<p class='caption-small'>Live data via Yahoo Finance. Educational tool — "
    "not investment advice.</p>", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================
st.title("📉 Portfolio Risk Engine")
st.markdown(
    "Multi-method **Value-at-Risk** & **Expected Shortfall** with a best-case "
    "upside view and regulatory backtesting — across any multi-asset portfolio."
)

if not run:
    st.info("👈 Set up your portfolio in the sidebar and hit **Run risk analysis**.")
    with st.expander("What this tool does", expanded=True):
        st.markdown(
            """
            - **Worst case:** estimates how much the portfolio could lose in a single
              day (VaR) and the average loss *beyond* that point (Expected Shortfall),
              using four independent methods to cross-check each other.
            - **Best case:** the same idea mirrored to the upside — *Value-at-Gain*
              and *Expected Gain* — so you see the full distribution of outcomes,
              not just the downside.
            - **Validation:** walks the VaR model forward day by day and runs the
              **Kupiec** and **Christoffersen** tests (the Basel III / FRTB standard)
              to check the model is actually well-calibrated.
            """
        )
    st.stop()

# --- Validation ------------------------------------------------------------
if len(tickers) < 2:
    st.error("Please enter at least two tickers."); st.stop()

with st.spinner("Fetching market data & computing risk…"):
    try:
        prices = fetch_prices(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No data for: {', '.join(missing)} — dropping them.")
        tickers = [t for t in tickers if t in prices.columns]
        weights = np.repeat(1 / len(tickers), len(tickers)) if weight_mode == "Equal weight" \
            else weights[[i for i, t in enumerate(tickers_raw.split(',')) if t.strip().upper() in tickers]]
        weights = weights / weights.sum()
    prices = prices[tickers]

    if prices.shape[0] < bt_window + 50:
        st.error(f"Only {prices.shape[0]} trading days available — need at least "
                 f"{bt_window + 50} for the chosen backtest window. Widen the date range "
                 "or shrink the window."); st.stop()

    asset_returns = prices.pct_change().dropna()
    portfolio_returns = asset_returns @ weights
    portfolio_returns.name = "portfolio"
    pv = notional

    ann_ret = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan

    # Full method comparison at the chosen confidence level
    rows = []
    for name, fn in METHODS.items():
        dv, de, uv, ue = fn(portfolio_returns, alpha)
        rows.append({"Method": name,
                     "VaR_$": dv * pv, "ES_$": de * pv,
                     "VaG_$": uv * pv, "EG_$": ue * pv,
                     "VaR_%": dv, "ES_%": de, "VaG_%": uv, "EG_%": ue})
    res = pd.DataFrame(rows).set_index("Method")

    # Backtest at the chosen level
    vf, rz, ex = rolling_var_backtest(portfolio_returns, alpha, int(bt_window), bt_method)
    lr_k, p_k, obs_exc, exp_exc = kupiec_pof(ex, alpha)
    lr_c, p_c = christoffersen_independence(ex)

# ============================================================================
# KPI ROW
# ============================================================================
st.subheader(f"Portfolio snapshot · {', '.join(tickers)}")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Annualised return", f"{ann_ret:.1%}")
k2.metric("Annualised volatility", f"{ann_vol:.1%}")
k3.metric("Sharpe (rf=0)", f"{sharpe:.2f}")
k4.metric(f"{conf_pct:g}% 1-day VaR", f"${res.loc['Historical','VaR_$']:,.0f}",
          help="Worst-case: loss not expected to be exceeded on a normal day.")
k5.metric(f"{conf_pct:g}% 1-day Value-at-Gain", f"${res.loc['Historical','VaG_$']:,.0f}",
          delta="upside", delta_color="normal",
          help="Best-case: gain you'd exceed only on the strongest days.")

st.caption(f"Based on {prices.shape[0]:,} trading days "
           f"({prices.index[0].date()} → {prices.index[-1].date()}) · "
           f"Notional ${pv:,.0f}")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Risk summary", "🔔 Distribution", "🧪 Backtest", "📄 Report"])

# ---- TAB 1: Risk summary ---------------------------------------------------
with tab1:
    left, right = st.columns([3, 2])

    with left:
        order = ["Parametric-N", "MonteCarlo", "Historical", "Parametric-t"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f"Worst case — VaR & ES ({conf_pct:g}%)",
            f"Best case — VaG & EG ({conf_pct:g}%)"))
        fig.add_trace(go.Bar(x=order, y=res.loc[order, "VaR_$"], name="VaR",
                             marker_color=C_DOWN), row=1, col=1)
        fig.add_trace(go.Bar(x=order, y=res.loc[order, "ES_$"], name="ES",
                             marker_color="#7A2E31"), row=1, col=1)
        fig.add_trace(go.Bar(x=order, y=res.loc[order, "VaG_$"], name="VaG",
                             marker_color=C_UP), row=1, col=2)
        fig.add_trace(go.Bar(x=order, y=res.loc[order, "EG_$"], name="EG",
                             marker_color="#2F6B45"), row=1, col=2)
        fig.update_layout(barmode="group", height=430, legend_title_text="",
                          margin=dict(t=50, b=10, l=10, r=10),
                          template="plotly_white")
        fig.update_yaxes(title_text="USD", tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Loss / gain by method** (at your confidence level)")
        show = res[["VaR_$", "ES_$", "VaG_$", "EG_$"]].copy()
        show.columns = ["VaR (loss)", "ES (loss)", "VaG (gain)", "EG (gain)"]
        st.dataframe(show.style.format("${:,.0f}"), use_container_width=True)
        spread = res.loc["Historical", "EG_$"] - res.loc["Historical", "ES_$"]
        st.markdown(
            f"<p class='caption-small'>The four methods should broadly agree; a large "
            f"gap for <b>Parametric-t</b> vs <b>Parametric-N</b> is the fat-tail effect — "
            f"the Gaussian model tends to understate real tail risk.</p>",
            unsafe_allow_html=True)

# ---- TAB 2: Distribution ---------------------------------------------------
with tab2:
    r = portfolio_returns.values
    dv, de, uv, ue = historical_tails(portfolio_returns, alpha)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=140, histnorm="probability density",
                               marker_color=C_BASE, opacity=0.65, name="Daily returns"))
    fig.add_trace(go.Histogram(x=r[r <= -dv], nbinsx=40, histnorm="probability density",
                               marker_color=C_DOWN, opacity=0.75, name="Downside tail"))
    fig.add_trace(go.Histogram(x=r[r >= uv], nbinsx=40, histnorm="probability density",
                               marker_color=C_UP, opacity=0.75, name="Upside tail"))
    # Fitted normal overlay
    xs = np.linspace(r.min(), r.max(), 400)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), r.std()),
                             line=dict(color="black", dash="dot"), name="Fitted normal"))
    fig.add_vline(x=-dv, line=dict(color=C_DOWN, dash="dash"),
                  annotation_text=f"VaR {dv:.2%}", annotation_position="top left")
    fig.add_vline(x=uv, line=dict(color=C_UP, dash="dash"),
                  annotation_text=f"VaG {uv:.2%}", annotation_position="top right")
    fig.update_layout(barmode="overlay", height=520, template="plotly_white",
                      xaxis_tickformat=".0%", xaxis_title="Daily return",
                      yaxis_title="Density", margin=dict(t=30, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.markdown(
        f"**Worst case (downside):** on the worst {(1-alpha):.1%} of days you'd expect to "
        f"lose more than **{dv:.2%}** (≈ **${dv*pv:,.0f}**), and when that happens the "
        f"average loss is **{de:.2%}** (≈ **${de*pv:,.0f}**).")
    c2.markdown(
        f"**Best case (upside):** on the best {(1-alpha):.1%} of days you'd expect to "
        f"gain more than **{uv:.2%}** (≈ **${uv*pv:,.0f}**), averaging **{ue:.2%}** "
        f"(≈ **${ue*pv:,.0f}**) in that top tail.")

# ---- TAB 3: Backtest -------------------------------------------------------
with tab3:
    dates = portfolio_returns.index[int(bt_window):]
    exc_mask = ex.astype(bool)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=rz, mode="lines", line=dict(color=C_GREY, width=0.8),
                             name="Daily return"))
    fig.add_trace(go.Scatter(x=dates, y=-vf, mode="lines", line=dict(color=C_DOWN, width=1.6),
                             name=f"{conf_pct:g}% VaR forecast"))
    fig.add_trace(go.Scatter(x=dates[exc_mask], y=rz[exc_mask], mode="markers",
                             marker=dict(color="darkred", size=7, line=dict(color="black", width=0.5)),
                             name=f"Exceptions ({int(exc_mask.sum())})"))
    fig.update_layout(height=480, template="plotly_white", yaxis_tickformat=".0%",
                      yaxis_title="Daily return", margin=dict(t=30, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    passed = (p_k > 0.05) and (p_c > 0.05)
    b1, b2, b3 = st.columns(3)
    b1.metric("Exceptions observed", f"{int(obs_exc)}", help=f"Expected ≈ {exp_exc:.1f}")
    b2.metric("Kupiec p-value", f"{p_k:.3f}" if not np.isnan(p_k) else "n/a")
    b3.metric("Christoffersen p-value", f"{p_c:.3f}" if not np.isnan(p_c) else "n/a")
    if passed:
        st.success("✅ **Well-calibrated** — both tests fail to reject the model "
                   "(p > 0.05). Breach frequency and timing look correct.")
    else:
        st.warning("⚠️ **Review** — a test rejects at p < 0.05. Breaches may be too "
                   "frequent or clustered in stress periods (consider EWMA/GARCH volatility).")
    st.caption(f"Method: {bt_method} · window: {int(bt_window)} days · "
               f"{len(ex):,} out-of-sample test days.")

# ---- TAB 4: Report ---------------------------------------------------------
with tab4:
    st.markdown(f"#### Portfolio risk report — 1-day horizon")
    st.markdown(f"**Assets:** {', '.join(tickers)}  \n"
                f"**Weights:** {', '.join(f'{t} {w:.0%}' for t, w in zip(tickers, weights))}  \n"
                f"**Notional:** ${pv:,.0f}  \n"
                f"**Period:** {prices.index[0].date()} → {prices.index[-1].date()}")
    report = res[["VaR_$", "ES_$", "VaG_$", "EG_$"]].copy()
    report.columns = ["VaR (loss $)", "ES (loss $)", "VaG (gain $)", "EG (gain $)"]
    st.dataframe(report.style.format("${:,.0f}"), use_container_width=True)
    st.markdown(
        f"- **Backtest verdict ({conf_pct:g}%):** "
        f"{'✅ well-calibrated' if (p_k>0.05 and p_c>0.05) else '⚠️ review model'} "
        f"(Kupiec p={p_k:.3f}, Christoffersen p={p_c:.3f}).\n"
        f"- **Regulatory note:** the 97.5% Expected Shortfall is the FRTB-mandated "
        f"risk measure under Basel III.\n"
        f"- **Reading it:** VaR/ES describe the downside you should plan for; "
        f"VaG/EG describe the upside you could reasonably capture — together they frame "
        f"the full risk/reward picture, not just the loss side."
    )
    csv = report.to_csv().encode()
    st.download_button("⬇ Download report (CSV)", csv, "risk_report.csv", "text/csv")

st.markdown("---")
st.markdown(
    "<p class='caption-small'>Portfolio Risk Engine · methods: Historical Simulation, "
    "Parametric (Gaussian & Student-t), Monte Carlo · backtests: Kupiec POF & "
    "Christoffersen independence · data: Yahoo Finance. For educational use only; "
    "not investment advice.</p>", unsafe_allow_html=True)
