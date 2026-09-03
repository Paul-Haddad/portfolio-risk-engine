"""Portfolio Optimizer page — Markowitz efficient frontier & optimal allocations."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import core as C

st.set_page_config(page_title="Optimizer · PortaRisk", page_icon="📊", layout="wide")
st.markdown("<style>.block-container{max-width:1200px;padding-top:2rem;}"
            "[data-testid='stMetricValue']{font-size:1.45rem;}</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("⚙️ Optimizer setup")
tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", "SPY, TLT, GLD, QQQ, EEM")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
a, b = st.sidebar.columns(2)
start = a.date_input("Start", pd.to_datetime("2015-01-01"))
end = b.date_input("End", pd.to_datetime("today"))
rf_pct = st.sidebar.number_input("Risk-free rate (%, annual)", 0.0, 10.0, 2.0, 0.25)
rf = rf_pct/100
n_random = st.sidebar.select_slider("Random portfolios to simulate",
                                    [1000, 2500, 5000, 10000], 5000)
run = st.sidebar.button("▶  Optimise portfolio", type="primary", use_container_width=True)

# ---------------------------------------------------------------- Header
st.title("📊 Portfolio Optimizer")
st.caption("Markowitz mean-variance optimisation: the efficient frontier and the optimal "
           "long-only allocations for your assets.")

if not run:
    st.info("👈 Enter your assets and hit **Optimise portfolio**.")
    st.stop()
if len(tickers) < 2:
    st.error("Enter at least two tickers."); st.stop()

with st.spinner("Fetching data & optimising…"):
    try:
        prices = C.fetch_prices(tuple(tickers), start, end)
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()
    have = [t for t in tickers if t in prices.columns]
    if len(have) < 2:
        st.error("Couldn't load enough of those tickers."); st.stop()
    if len(have) < len(tickers):
        st.warning(f"No data for {', '.join(set(tickers)-set(have))} — dropping them.")
        tickers = have
    prices = prices[tickers]
    asset_ret = prices.pct_change().dropna()
    mu, Sigma = C.annualised_moments(asset_ret)

    w_ms = C.max_sharpe(mu, Sigma, rf)
    w_mv = C.min_variance(mu, Sigma)
    eq = np.repeat(1/len(tickers), len(tickers))
    fvol, fret = C.efficient_frontier(mu, Sigma, 45)
    rvol, rret, rsharpe, _ = C.random_portfolios(mu, Sigma, int(n_random), rf)

    def perf(w): return C.portfolio_perf(w, mu, Sigma, rf)
    ms_ret, ms_vol, ms_sh = perf(w_ms)
    mv_ret, mv_vol, mv_sh = perf(w_mv)
    eq_ret, eq_vol, eq_sh = perf(eq)

# ---------------------------------------------------------------- KPIs
st.subheader(f"Optimal portfolio · {', '.join(tickers)}")
k = st.columns(4)
k[0].metric("Max-Sharpe return", f"{ms_ret:.1%}")
k[1].metric("Max-Sharpe volatility", f"{ms_vol:.1%}")
k[2].metric("Max Sharpe ratio", f"{ms_sh:.2f}",
            delta=f"{ms_sh-eq_sh:+.2f} vs equal-weight")
k[3].metric("Min-variance volatility", f"{mv_vol:.1%}")
st.caption(f"{prices.shape[0]:,} trading days · {prices.index[0].date()} → "
           f"{prices.index[-1].date()} · risk-free {rf_pct:.2f}%")

t1, t2, t3 = st.tabs(["🌐 Efficient frontier", "⚖️ Allocations", "🔗 Correlations"])

# ---------------------------------------------------------------- Tab 1
with t1:
    fig = go.Figure()
    fig.add_scatter(x=rvol, y=rret, mode="markers",
                    marker=dict(size=5, color=rsharpe, colorscale="Viridis",
                                colorbar=dict(title="Sharpe"), opacity=0.55),
                    name="Random portfolios", hovertemplate="vol %{x:.1%}<br>ret %{y:.1%}")
    fig.add_scatter(x=fvol, y=fret, mode="lines", line=dict(color="#E6EAF1", width=2.5),
                    name="Efficient frontier")
    fig.add_scatter(x=[ms_vol], y=[ms_ret], mode="markers",
                    marker=dict(symbol="star", size=18, color=C.C_UP,
                                line=dict(color="white", width=1)), name="Max Sharpe")
    fig.add_scatter(x=[mv_vol], y=[mv_ret], mode="markers",
                    marker=dict(symbol="diamond", size=14, color=C.C_ACCENT,
                                line=dict(color="white", width=1)), name="Min variance")
    fig.add_scatter(x=[eq_vol], y=[eq_ret], mode="markers",
                    marker=dict(symbol="x", size=13, color=C.C_DOWN), name="Equal weight")
    fig.update_layout(height=540, template="plotly_dark",
                      xaxis_title="Annualised volatility (risk)", yaxis_title="Annualised return",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each dot is a random long-only mix; the curve is the best achievable return "
               "for each level of risk. The ⭐ is the highest-Sharpe portfolio; the ◆ is the "
               "lowest-risk one. Your equal-weight mix (✕) sits inside the frontier — the gap "
               "to the curve is the improvement optimisation buys you.")

# ---------------------------------------------------------------- Tab 2
with t2:
    which = st.radio("Show weights for", ["Max Sharpe", "Min variance", "Equal weight"],
                     horizontal=True)
    w = {"Max Sharpe": w_ms, "Min variance": w_mv, "Equal weight": eq}[which]
    c1, c2 = st.columns([3, 2])
    with c1:
        order = np.argsort(w)[::-1]
        fig = go.Figure(go.Bar(x=np.array(tickers)[order], y=w[order],
                               marker_color=C.C_ACCENT, text=[f"{x:.0%}" for x in w[order]],
                               textposition="outside"))
        fig.update_layout(height=360, template="plotly_dark", yaxis_tickformat=".0%",
                          title=f"{which} allocation", margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        comp = pd.DataFrame({
            "Max Sharpe": w_ms, "Min variance": w_mv, "Equal weight": eq}, index=tickers)
        st.dataframe(comp.style.format("{:.1%}"), use_container_width=True)
        summ = pd.DataFrame({
            "Return": [ms_ret, mv_ret, eq_ret],
            "Volatility": [ms_vol, mv_vol, eq_vol],
            "Sharpe": [ms_sh, mv_sh, eq_sh]},
            index=["Max Sharpe", "Min variance", "Equal weight"])
        st.dataframe(summ.style.format({"Return": "{:.1%}", "Volatility": "{:.1%}",
                                        "Sharpe": "{:.2f}"}), use_container_width=True)
    st.download_button("⬇ Download allocations (CSV)", comp.to_csv().encode(),
                       "optimal_weights.csv", "text/csv")

# ---------------------------------------------------------------- Tab 3
with t3:
    corr = asset_ret.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=420, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Low or negative correlations are what let diversification cut risk — the "
               "optimiser leans on assets that zig when others zag.")

with st.expander("📖 Methodology — how the optimiser works"):
    st.markdown(
        """
        - **Inputs:** each asset's annualised expected return (from its historical mean) and
          the annualised **covariance matrix** (how the assets move together).
        - **Efficient frontier:** for every target return, the optimiser finds the long-only
          mix with the *lowest* variance (solved with sequential quadratic programming, weights
          ≥ 0 and summing to 100%). The result is the classic Markowitz curve.
        - **Max-Sharpe portfolio (⭐):** the mix with the best return per unit of risk above the
          risk-free rate — the tangency portfolio.
        - **Min-variance portfolio (◆):** the lowest-risk mix, ignoring return.
        - **Caveat:** optimisation uses *past* returns as the estimate of the future, so
          treat it as a disciplined starting point, not a guarantee. Pairs naturally with the
          **Risk Engine**, which stress-tests the tails of whatever mix you choose.
        """)

st.markdown("---")
st.caption("Educational tool — not investment advice.")
