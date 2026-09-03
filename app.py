"""
PortaRisk — Portfolio Analytics Suite  (Home)
=============================================
A two-tool suite for multi-asset portfolios:
  • Risk Engine    — VaR / Expected Shortfall + upside, with regulatory backtests
  • Optimizer      — Markowitz efficient frontier & optimal allocations

Entry point for Streamlit multipage app. Author: Paul Haddad.
Deploy: push repo to GitHub -> share.streamlit.io -> main file = app.py
"""
import streamlit as st

st.set_page_config(page_title="PortaRisk — Portfolio Analytics",
                   page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1180px;}
      .hero {font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin-bottom: .3rem;}
      .sub {color:#AAB4C4; font-size: 1.05rem; margin-bottom: 1.6rem;}
      .card {background:#171A23; border:1px solid #262B36; border-radius:14px;
             padding:1.3rem 1.4rem; height:100%;}
      .card h3 {margin:0 0 .4rem 0;}
      .pill {display:inline-block; background:#1E2A38; color:#7FB6EA;
             border-radius:999px; padding:2px 10px; font-size:.75rem; margin-bottom:.6rem;}
      .muted {color:#8899AA; font-size:.85rem;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="hero">📈 PortaRisk</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">A portfolio analytics suite — measure the risk of any '
            'multi-asset portfolio, then build the optimal one.</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="card">
          <span class="pill">TOOL 1</span>
          <h3>📉 Risk Engine</h3>
          <p>Estimates <b>Value-at-Risk</b> and <b>Expected Shortfall</b> four independent
          ways (Historical, Gaussian, Student-t, Monte Carlo), mirrors them into a
          <b>best-case upside</b> view, and validates the model with the
          <b>Kupiec</b> and <b>Christoffersen</b> backtests used under Basel III / FRTB.</p>
          <p class="muted">How much could this portfolio lose — or gain — on a given day,
          and is the model actually trustworthy?</p>
        </div>
        """, unsafe_allow_html=True)
with c2:
    st.markdown(
        """
        <div class="card">
          <span class="pill">TOOL 2</span>
          <h3>📊 Portfolio Optimizer</h3>
          <p>Runs <b>Markowitz mean-variance optimisation</b>: plots the
          <b>efficient frontier</b>, finds the <b>maximum-Sharpe</b> and
          <b>minimum-variance</b> portfolios, and shows how your current mix compares —
          with the correlation structure behind it all.</p>
          <p class="muted">Given these assets, what mix gives the best return for the
          risk taken?</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.info("Use the sidebar (top-left ») to open the **Risk Engine** or the **Optimizer**.")

with st.expander("How the two tools fit together"):
    st.markdown(
        "The Optimizer answers *what should I hold?* — it searches every long-only mix "
        "of your assets for the best risk/return trade-off. The Risk Engine answers "
        "*how dangerous is what I hold?* — it stress-tests a specific portfolio's tails. "
        "Used together they mirror a real risk/quant desk: **construct** an optimal "
        "portfolio, then **measure** the tail risk you've actually taken on.")

st.markdown("---")
st.markdown(
    '<p class="muted">Built by Paul Haddad · methods: Historical Simulation, Parametric '
    '(Gaussian & Student-t), Monte Carlo, Markowitz MVO · data: Yahoo Finance · '
    'Educational tool, not investment advice.</p>', unsafe_allow_html=True)
