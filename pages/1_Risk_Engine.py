"""Risk Engine page — VaR / ES + upside, distribution, backtest, diagnostics."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import streamlit as st

import core as C

st.set_page_config(page_title="Risk Engine · PortaRisk", layout="wide")
st.markdown("<style>.block-container{max-width:1200px;padding-top:2rem;}"
            "[data-testid='stMetricValue']{font-size:1.45rem;}</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("Portfolio setup")
tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", "SPY, TLT, GLD, QQQ, EEM")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

mode = st.sidebar.radio("Weights", ["Equal weight", "Custom"], horizontal=True)
if mode == "Custom" and tickers:
    raw = [st.sidebar.number_input(t, 0.0, value=round(1/len(tickers), 2), step=0.05,
                                   key=f"rw_{t}") for t in tickers]
    w = np.array(raw, float)
    weights = w/w.sum() if w.sum() > 0 else np.repeat(1/len(tickers), len(tickers))
else:
    weights = np.repeat(1/len(tickers), len(tickers)) if tickers else np.array([])

a, b = st.sidebar.columns(2)
start = a.date_input("Start", pd.to_datetime("2015-01-01"))
end = b.date_input("End", pd.to_datetime("today"))
notional = st.sidebar.number_input("Portfolio value (USD)", 1000, value=1_000_000, step=50_000, format="%d")
conf = st.sidebar.select_slider("Confidence level", [90, 95, 97.5, 99, 99.5], 99)
alpha = conf/100
with st.sidebar.expander("Advanced"):
    mc = st.number_input("Monte Carlo simulations", 1000, 200000, 50000, 5000)
    bt_window = st.number_input("Backtest window (days)", 100, 1000, 500, 50)
    bt_method = st.selectbox("Backtest method", ["historical", "parametric"])
run = st.sidebar.button("Run risk analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------- Header
st.title("Risk Engine")
st.caption("Multi-method Value-at-Risk & Expected Shortfall, with a best-case upside "
           "view and regulatory backtesting.")

if not run:
    st.info("Set up your portfolio in the sidebar, then run the analysis.")
    st.stop()
if len(tickers) < 2:
    st.error("Enter at least two tickers."); st.stop()

with st.spinner("Fetching data & computing risk…"):
    try:
        prices = C.fetch_prices(tuple(tickers), start, end)
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()
    have = [t for t in tickers if t in prices.columns]
    if len(have) < 2:
        st.error("Couldn't load enough of those tickers."); st.stop()
    if len(have) < len(tickers):
        st.warning(f"No data for {', '.join(set(tickers)-set(have))} — dropping them.")
        weights = np.repeat(1/len(have), len(have)); tickers = have
    prices = prices[tickers]
    if prices.shape[0] < bt_window + 50:
        st.error(f"Only {prices.shape[0]} days available; need at least {bt_window+50}. "
                 "Widen the dates or shrink the backtest window."); st.stop()

    asset_ret, port = C.portfolio_series(prices, weights)
    pv = notional
    ann_ret, ann_vol, sharpe = C.annualised_stats(port)
    tbl = C.tails_by_method(port, alpha, mc)
    dollar = (tbl * pv)
    vf, rz, ex = C.rolling_var_backtest(port, alpha, int(bt_window), bt_method)
    _, p_k, obs, exp = C.kupiec_pof(ex, alpha)
    _, p_c = C.christoffersen_independence(ex)

# ---------------------------------------------------------------- KPIs
st.subheader(f"Snapshot · {', '.join(tickers)}")
k = st.columns(5)
k[0].metric("Annualised return", f"{ann_ret:.1%}")
k[1].metric("Annualised volatility", f"{ann_vol:.1%}")
k[2].metric("Sharpe (rf=0)", f"{sharpe:.2f}")
k[3].metric(f"{conf:g}% 1-day VaR", f"${dollar.loc['Historical','VaR']:,.0f}",
            help="Worst case: loss not exceeded on a normal day.")
k[4].metric(f"{conf:g}% 1-day VaG", f"${dollar.loc['Historical','VaG']:,.0f}",
            delta="upside", help="Best case: gain exceeded only on the strongest days.")
st.caption(f"{prices.shape[0]:,} trading days · {prices.index[0].date()} → "
           f"{prices.index[-1].date()} · notional ${pv:,.0f}")

t1, t2, t3, t4 = st.tabs(["Risk summary", "Distribution", "Backtest", "Diagnostics"])

# ---------------------------------------------------------------- Tab 1
with t1:
    order = ["Historical", "Parametric-N", "Parametric-t", "MonteCarlo"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=order, y=dollar.loc[order, "VaR"], name="VaR", marker_color=C.C_DOWN)
        fig.add_bar(x=order, y=dollar.loc[order, "ES"], name="ES", marker_color="#9E3B3D")
        fig.update_layout(title=f"Worst case — VaR & ES ({conf:g}%)", barmode="group",
                          height=380, yaxis_tickformat="$,.0f",
                          margin=dict(t=50, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_bar(x=order, y=dollar.loc[order, "VaG"], name="VaG", marker_color=C.C_UP)
        fig.add_bar(x=order, y=dollar.loc[order, "EG"], name="EG", marker_color="#3C7A38")
        fig.update_layout(title=f"Best case — VaG & EG ({conf:g}%)", barmode="group",
                          height=380, yaxis_tickformat="$,.0f",
                          margin=dict(t=50, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    show = dollar.copy(); show.columns = ["VaR (loss)", "ES (loss)", "VaG (gain)", "EG (gain)"]
    st.dataframe(show.style.format("${:,.0f}"), use_container_width=True)
    st.caption("The four methods should broadly agree. A larger Parametric-t figure than "
               "Parametric-N is the fat-tail effect — the Gaussian model understates real tail risk.")

# ---------------------------------------------------------------- Tab 2
with t2:
    r = port.values
    dv, de, uv, ue = C.historical_tails(port, alpha)
    fig = go.Figure()
    fig.add_histogram(x=r, nbinsx=140, histnorm="probability density",
                      marker_color=C.C_BASE, opacity=0.55, name="Daily returns")
    fig.add_histogram(x=r[r <= -dv], nbinsx=40, histnorm="probability density",
                      marker_color=C.C_DOWN, opacity=0.8, name="Downside tail")
    fig.add_histogram(x=r[r >= uv], nbinsx=40, histnorm="probability density",
                      marker_color=C.C_UP, opacity=0.8, name="Upside tail")
    xs = np.linspace(r.min(), r.max(), 400)
    fig.add_scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), r.std()),
                    line=dict(color="#888888", dash="dot"), name="Fitted normal")
    fig.add_vline(x=-dv, line=dict(color=C.C_DOWN, dash="dash"),
                  annotation_text=f"VaR {dv:.2%}", annotation_position="top left")
    fig.add_vline(x=uv, line=dict(color=C.C_UP, dash="dash"),
                  annotation_text=f"VaG {uv:.2%}", annotation_position="top right")
    fig.update_layout(barmode="overlay", height=500, xaxis_tickformat=".0%",
                      xaxis_title="Daily return", yaxis_title="Density",
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)
    c = st.columns(2)
    c[0].markdown(f"**Worst case:** on the worst {1-alpha:.1%} of days you'd expect to lose "
                  f"more than **{dv:.2%}** (≈ ${dv*pv:,.0f}); the average loss beyond that is "
                  f"**{de:.2%}** (≈ ${de*pv:,.0f}).")
    c[1].markdown(f"**Best case:** on the best {1-alpha:.1%} of days you'd expect to gain more "
                  f"than **{uv:.2%}** (≈ ${uv*pv:,.0f}), averaging **{ue:.2%}** "
                  f"(≈ ${ue*pv:,.0f}) in that top tail.")

# ---------------------------------------------------------------- Tab 3
with t3:
    dates = port.index[int(bt_window):]; m = ex.astype(bool)
    fig = go.Figure()
    fig.add_scatter(x=dates, y=rz, mode="lines", line=dict(color=C.C_GREY, width=0.8),
                    name="Daily return")
    fig.add_scatter(x=dates, y=-vf, mode="lines", line=dict(color=C.C_DOWN, width=1.6),
                    name=f"{conf:g}% VaR forecast")
    fig.add_scatter(x=dates[m], y=rz[m], mode="markers",
                    marker=dict(color="#C0392B", size=7, line=dict(color="#7A2E31", width=.5)),
                    name=f"Exceptions ({int(m.sum())})")
    fig.update_layout(height=460, yaxis_tickformat=".0%", yaxis_title="Daily return",
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)
    passed = (p_k > 0.05) and (p_c > 0.05)
    m3 = st.columns(3)
    m3[0].metric("Exceptions", f"{int(obs)}", help=f"Expected ≈ {exp:.1f}")
    m3[1].metric("Kupiec p", f"{p_k:.3f}" if not np.isnan(p_k) else "n/a")
    m3[2].metric("Christoffersen p", f"{p_c:.3f}" if not np.isnan(p_c) else "n/a")
    (st.success if passed else st.warning)(
        "Well-calibrated — both tests fail to reject the model (p > 0.05)." if passed
        else "Review — a test rejects at p < 0.05: breaches may be too frequent or clustered.")

# ---------------------------------------------------------------- Tab 4
with t4:
    st.markdown("**Correlation between assets**")
    corr = asset_ret.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Contribution to portfolio risk**")
        rc = C.risk_contribution(asset_ret, weights).sort_values()
        fig = go.Figure(go.Bar(x=rc.values, y=rc.index, orientation="h", marker_color=C.C_ACCENT))
        fig.update_layout(height=320, xaxis_tickformat=".0%",
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Share of total portfolio variance from each asset — not the same as its weight.")
    with d2:
        st.markdown("**Cumulative return & drawdown**")
        curve, dd, mdd = C.max_drawdown(port)
        fig = go.Figure()
        fig.add_scatter(x=curve.index, y=curve.values, line=dict(color=C.C_UP), name="Growth of $1")
        fig.add_scatter(x=dd.index, y=dd.values, line=dict(color=C.C_DOWN), name="Drawdown", yaxis="y2")
        fig.update_layout(height=320, yaxis=dict(title="Growth of $1"),
                          yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
                          margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Max drawdown over the period: **{mdd:.1%}**.")

# ---------------------------------------------------------------- Methodology
with st.expander("Methodology — how each number is computed"):
    st.markdown(
        """
        - **VaR (Value-at-Risk):** the loss your portfolio is *not* expected to exceed on a
          normal day, at the chosen confidence level (e.g. 99% → the worst 1-in-100 day).
        - **ES (Expected Shortfall):** the *average* loss on the days worse than VaR — it
          captures how bad the tail really is. The **97.5% ES** is the FRTB-mandated
          regulatory risk measure under Basel III.
        - **VaG / EG (upside mirror):** the same idea flipped — the gain you'd exceed only on
          your strongest days, and the average gain in that top tail.
        - **Four methods:** Historical (empirical), Parametric-N (Gaussian), Parametric-t
          (Student-t, fat tails), and Monte Carlo (simulation). Cross-checking them guards
          against any single model's blind spot.
        - **Backtests:** the model is walked forward day by day; Kupiec checks the breach
          *frequency* is right, Christoffersen checks breaches don't *cluster*. p > 0.05 on
          both means well-calibrated.
        """)

st.divider()
st.caption("Educational tool — not investment advice.")
