"""
PortaRisk — Portfolio Analytics Suite  (Home)
=============================================
Two tools for multi-asset portfolios:
  - Risk Engine  — VaR / Expected Shortfall + upside, with regulatory backtests
  - Optimizer    — Markowitz efficient frontier & optimal allocations

Entry point for the Streamlit multipage app. Author: Paul Haddad.
Deploy: push repo to GitHub -> share.streamlit.io -> main file = app.py
"""
import streamlit as st

st.set_page_config(page_title="PortaRisk — Portfolio Analytics", layout="wide")

st.markdown(
    "<style>.block-container{max-width:1150px;padding-top:2.2rem;}"
    "h1{letter-spacing:-0.5px;}</style>",
    unsafe_allow_html=True,
)

st.title("PortaRisk")
st.caption("A portfolio analytics suite — measure the risk of any multi-asset "
           "portfolio, then build the optimal one.")
st.write("")

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.caption("TOOL 1")
        st.subheader("Risk Engine")
        st.write(
            "Estimates **Value-at-Risk** and **Expected Shortfall** four independent "
            "ways (Historical, Gaussian, Student-t, Monte Carlo), mirrors them into a "
            "**best-case upside** view, and validates the model with the **Kupiec** and "
            "**Christoffersen** backtests used under Basel III / FRTB.")
        st.caption("How much could this portfolio lose — or gain — on a given day, "
                   "and is the model actually trustworthy?")
with c2:
    with st.container(border=True):
        st.caption("TOOL 2")
        st.subheader("Portfolio Optimizer")
        st.write(
            "Runs **Markowitz mean-variance optimisation**: plots the **efficient "
            "frontier**, finds the **maximum-Sharpe** and **minimum-variance** "
            "portfolios, and shows how your current mix compares — with the "
            "correlation structure behind it all.")
        st.caption("Given these assets, what mix gives the best return for the "
                   "risk taken?")

st.write("")
st.info("Open the **Risk Engine** or the **Optimizer** from the sidebar on the left.")

with st.expander("How the two tools fit together"):
    st.write(
        "The Optimizer answers *what should I hold?* — it searches every long-only mix "
        "of your assets for the best risk/return trade-off. The Risk Engine answers "
        "*how dangerous is what I hold?* — it stress-tests a specific portfolio's tails. "
        "Used together they mirror a real risk desk: construct an optimal portfolio, "
        "then measure the tail risk you've taken on.")

st.divider()
st.caption(
    "Built by Paul Haddad · methods: Historical Simulation, Parametric (Gaussian & "
    "Student-t), Monte Carlo, Markowitz MVO · data: Yahoo Finance · "
    "Educational tool, not investment advice.")
