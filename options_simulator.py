"""
Monte Carlo Options Pricing Simulator — Streamlit + Plotly Edition
Run:  streamlit run options_simulator.py
Dependencies: streamlit, numpy, scipy, plotly
"""

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Options Pricer",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stApp {
        background-color: #f7f7f5;
        color: #111111;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #111111 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #111111 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    section[data-testid="stSidebar"] input {
        background: #f7f7f5 !important;
        border: 1px solid #cccccc !important;
        border-radius: 0 !important;
        color: #111111 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.95rem !important;
    }
    section[data-testid="stSidebar"] input:focus {
        border-color: #111111 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #888888 !important;
        font-size: 0.75rem !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1.5px solid #111111;
        border-radius: 0 !important;
        padding: 20px 24px !important;
    }
    div[data-testid="stMetric"] label {
        color: #666666 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111111 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Headers */
    h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
        color: #111111 !important;
        font-size: 2.8rem !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        margin-bottom: 0 !important;
        -webkit-text-fill-color: #111111 !important;
        background: none !important;
        text-align: left !important;
    }
    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #111111 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 1.5px solid #111111 !important;
        margin: 1.5rem 0 !important;
    }

    /* Expander */
    details {
        border: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        background: #ffffff !important;
    }
    summary {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #111111 !important;
        padding: 12px 16px !important;
    }

    /* Table */
    table {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }
    th {
        background: #111111 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        text-align: left !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-size: 0.72rem !important;
    }
    td {
        padding: 7px 12px !important;
        border-bottom: 1px solid #e5e5e5 !important;
        color: #111111 !important;
    }
    tr:nth-child(even) td {
        background: #f7f7f5 !important;
    }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888888;
        border-bottom: 1px solid #dddddd;
        padding-bottom: 6px;
        margin-bottom: 12px;
        margin-top: 20px;
    }

    /* Subtitle */
    .subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        color: #666666;
        font-weight: 300;
        margin-top: 4px;
        margin-bottom: 1.5rem;
        letter-spacing: 0.01em;
    }

    /* Chart label */
    .chart-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #111111;
        border-left: 3px solid #111111;
        padding-left: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Sidebar inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parameters")
    st.markdown('<p class="section-label">Market Data</p>', unsafe_allow_html=True)

    S     = st.number_input("Stock Price (S)",        min_value=1.0,  max_value=10000.0, value=100.0, step=1.0,   format="%.2f")
    K     = st.number_input("Strike Price (K)",       min_value=1.0,  max_value=10000.0, value=105.0, step=1.0,   format="%.2f")
    r     = st.number_input("Risk-Free Rate (r)",     min_value=0.0,  max_value=1.0,     value=0.05,  step=0.005, format="%.4f")
    sigma = st.number_input("Volatility (σ)",         min_value=0.01, max_value=5.0,     value=0.20,  step=0.01,  format="%.4f")

    st.markdown('<p class="section-label">Simulation</p>', unsafe_allow_html=True)

    T = st.number_input("Time to Expiration (years)", min_value=0.01, max_value=10.0,    value=1.0,   step=0.25,  format="%.2f")
    N = st.number_input("Simulations (N)",            min_value=100,  max_value=500000,  value=10000, step=1000)

    st.markdown("---")
    st.markdown(
        "<p style='color:#aaaaaa; font-size:0.72rem; font-family:IBM Plex Sans; letter-spacing:0.03em;'>"
        "Reruns automatically on input change.</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
#  Core math — Black-Scholes
# ──────────────────────────────────────────────
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ──────────────────────────────────────────────
#  Core math — Monte Carlo GBM
# ──────────────────────────────────────────────
def monte_carlo_call(S, K, T, r, sigma, N, n_steps=252):
    dt = T / n_steps
    Z = np.random.standard_normal((int(N), n_steps))
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    price_paths = S * np.exp(np.hstack([np.zeros((int(N), 1)), log_paths]))
    S_T = price_paths[:, -1]
    payoffs = np.maximum(S_T - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N)
    return mc_price, std_error, price_paths, payoffs


# ──────────────────────────────────────────────
#  Run simulation
# ──────────────────────────────────────────────
mc_price, std_error, price_paths, payoffs = monte_carlo_call(S, K, T, r, sigma, int(N))
bs_price   = black_scholes_call(S, K, T, r, sigma)
difference = mc_price - bs_price


# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────
st.markdown("# Monte Carlo Options Pricer")
st.markdown(
    '<p class="subtitle">European Call Option &nbsp;&middot;&nbsp; Geometric Brownian Motion &nbsp;&middot;&nbsp; Black-Scholes Benchmark</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ──────────────────────────────────────────────
#  Metric cards
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Monte Carlo Price",    value=f"${mc_price:,.4f}")
with col2:
    st.metric(label="Black-Scholes Price",  value=f"${bs_price:,.4f}")
with col3:
    st.metric(label="Difference (MC - BS)", value=f"${difference:,.4f}", delta=f"{difference:+.4f}", delta_color="normal")
with col4:
    st.metric(label="MC Std Error",         value=f"${std_error:,.4f}")

st.markdown("---")


# ──────────────────────────────────────────────
#  Plotly theming — light finance dashboard
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans, sans-serif", color="#111111", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(
        gridcolor="#eeeeee",
        linecolor="#111111",
        linewidth=1.5,
        tickfont=dict(family="IBM Plex Mono", size=10, color="#555555"),
        title_font=dict(family="IBM Plex Sans", size=10, color="#555555"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#eeeeee",
        linecolor="#111111",
        linewidth=1.5,
        tickfont=dict(family="IBM Plex Mono", size=10, color="#555555"),
        title_font=dict(family="IBM Plex Sans", size=10, color="#555555"),
        zeroline=False,
    ),
)


# ──────────────────────────────────────────────
#  Charts
# ──────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<p class="chart-label">Simulated Price Paths (first 100)</p>', unsafe_allow_html=True)

    n_display = min(100, int(N))
    t_axis    = np.linspace(0, T, price_paths.shape[1])
    fig_paths = go.Figure()

    for i in range(n_display):
        fig_paths.add_trace(go.Scatter(
            x=t_axis,
            y=price_paths[i],
            mode="lines",
            line=dict(width=0.6, color="rgba(30, 80, 180, 0.18)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig_paths.add_hline(
        y=K, line_dash="dash", line_color="#cc2200", line_width=1.5,
        annotation_text=f"Strike  K = {K}",
        annotation_position="top left",
        annotation_font=dict(color="#cc2200", size=10, family="IBM Plex Mono"),
    )
    fig_paths.add_hline(
        y=S, line_dash="dot", line_color="#111111", line_width=1.2,
        annotation_text=f"S0 = {S}",
        annotation_position="bottom left",
        annotation_font=dict(color="#111111", size=10, family="IBM Plex Mono"),
    )
    fig_paths.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Time (years)", yaxis_title="Stock Price ($)")
    st.plotly_chart(fig_paths, use_container_width=True, theme=None)

with chart_col2:
    st.markdown('<p class="chart-label">Payoff Distribution at Expiration</p>', unsafe_allow_html=True)

    mean_payoff = np.mean(payoffs)
    itm_pct     = np.mean(payoffs > 0) * 100
    fig_hist    = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=payoffs,
        nbinsx=80,
        marker=dict(
            color="rgba(30, 80, 180, 0.55)",
            line=dict(color="#1e50b4", width=0.3),
        ),
        hovertemplate="Payoff: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
        name="Payoffs",
    ))
    fig_hist.add_vline(
        x=mean_payoff, line_dash="dash", line_color="#cc2200", line_width=1.5,
        annotation_text=f"Mean  ${mean_payoff:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#cc2200", size=10, family="IBM Plex Mono"),
    )
    fig_hist.add_annotation(
        x=0.97, y=0.94, xref="paper", yref="paper",
        text=f"ITM: {itm_pct:.1f}%    OTM: {100 - itm_pct:.1f}%",
        showarrow=False,
        font=dict(color="#555555", size=10, family="IBM Plex Mono"),
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=6,
        xanchor="right",
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Call Payoff ($)", yaxis_title="Frequency", bargap=0.02)
    st.plotly_chart(fig_hist, use_container_width=True, theme=None)


# ──────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────
st.markdown("---")
with st.expander("About this simulation"):
    st.markdown("""
**Model:** Geometric Brownian Motion (GBM)

**Monte Carlo method:**
- Simulate N stock-price paths over 252 daily steps
- Payoff at expiration: max(S_T - K, 0)
- Discounted price: average of all payoffs x e^(-rT)

**Black-Scholes closed-form** is used as benchmark.
""")
    st.table({
        "Parameter": ["Stock Price (S)", "Strike Price (K)", "Risk-Free Rate (r)", "Volatility (σ)", "Time (T)", "Simulations (N)"],
        "Value":     [S, K, r, sigma, T, int(N)]
    })
