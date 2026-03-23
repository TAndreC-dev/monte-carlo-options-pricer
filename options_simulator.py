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
#  Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Options Pricer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }
    h1, h2, h3 { color: #e0e7ff !important; }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem !important;
        margin-bottom: 0 !important;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    hr { border-color: rgba(99, 102, 241, 0.15) !important; }
    section[data-testid="stSidebar"] label {
        color: #c7d2fe !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] input {
        background: rgba(30, 30, 60, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #e0e7ff !important;
        border-radius: 8px !important;
    }
    .section-header {
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Sidebar inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown('<p class="section-header">Market Data</p>', unsafe_allow_html=True)

    S = st.number_input("Stock Price (S)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=10000.0, value=105.0, step=1.0, format="%.2f")
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=5.0, value=0.20, step=0.01, format="%.4f")

    st.markdown('<p class="section-header">Simulation</p>', unsafe_allow_html=True)

    T = st.number_input("Time to Expiration (T, years)", min_value=0.01, max_value=10.0, value=1.0, step=0.25, format="%.2f")
    N = st.number_input("Number of Simulations (N)", min_value=100, max_value=500000, value=10000, step=1000)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "Simulation reruns automatically<br>when any input changes.</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
#  Core math
# ──────────────────────────────────────────────
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


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
bs_price = black_scholes_call(S, K, T, r, sigma)
difference = mc_price - bs_price

# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────
st.markdown("# Monte Carlo Options Pricer")
st.markdown('<p class="subtitle">European Call · Geometric Brownian Motion · Black-Scholes Benchmark</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Metric cards
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Monte Carlo Price", value=f"${mc_price:,.4f}")
with col2:
    st.metric(label="Black-Scholes Price", value=f"${bs_price:,.4f}")
with col3:
    st.metric(label="Difference (MC − BS)", value=f"${difference:,.4f}", delta=f"{difference:+.4f}", delta_color="normal")
with col4:
    st.metric(label="MC Std Error", value=f"${std_error:,.4f}")

st.markdown("---")

# ──────────────────────────────────────────────
#  Plotly theming
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,15,26,0.0)",
    plot_bgcolor="rgba(20,20,40,0.4)",
    font=dict(family="Inter, system-ui, sans-serif", color="#c7d2fe"),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
)

# ──────────────────────────────────────────────
#  Charts
# ──────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📊 Simulated Price Paths (first 100)")
    n_display = min(100, int(N))
    t_axis = np.linspace(0, T, price_paths.shape[1])
    fig_paths = go.Figure()
    colors = [f"hsla({210 + i * 1.5}, 80%, {55 + (i % 20)}%, 0.25)" for i in range(n_display)]
    for i in range(n_display):
        fig_paths.add_trace(go.Scatter(x=t_axis, y=price_paths[i], mode="lines",
                                       line=dict(width=0.8, color=colors[i]), showlegend=False, hoverinfo="skip"))
    fig_paths.add_hline(y=K, line_dash="dash", line_color="#f472b6", line_width=2,
                        annotation_text=f"Strike K = {K}", annotation_position="top left",
                        annotation_font=dict(color="#f472b6", size=12))
    fig_paths.add_hline(y=S, line_dash="dot", line_color="#34d399", line_width=1.5,
                        annotation_text=f"S0 = {S}", annotation_position="bottom left",
                        annotation_font=dict(color="#34d399", size=11))
    fig_paths.update_layout(**PLOTLY_LAYOUT, height=500, xaxis_title="Time (years)", yaxis_title="Stock Price ($)")
    st.plotly_chart(fig_paths, use_container_width=True, theme=None)

with chart_col2:
    st.markdown("### 📈 Payoff Distribution at Expiration")
    mean_payoff = np.mean(payoffs)
    itm_pct = np.mean(payoffs > 0) * 100
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=payoffs, nbinsx=80,
                                    marker=dict(color="rgba(129, 140, 248, 0.55)",
                                                line=dict(color="rgba(165, 180, 252, 0.6)", width=0.5)),
                                    hovertemplate="Payoff: $%{x:,.2f}<br>Count: %{y}<extra></extra>", name="Payoffs"))
    fig_hist.add_vline(x=mean_payoff, line_dash="dash", line_color="#fbbf24", line_width=2,
                       annotation_text=f"Mean = ${mean_payoff:,.2f}", annotation_position="top right",
                       annotation_font=dict(color="#fbbf24", size=12))
    fig_hist.add_annotation(x=0.98, y=0.95, xref="paper", yref="paper",
                             text=f"ITM: {itm_pct:.1f}%  |  OTM: {100 - itm_pct:.1f}%",
                             showarrow=False, font=dict(color="#94a3b8", size=11),
                             bgcolor="rgba(15,15,26,0.7)", bordercolor="rgba(99,102,241,0.2)",
                             borderwidth=1, borderpad=6, xanchor="right")
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=500, xaxis_title="Call Payoff ($)", yaxis_title="Frequency", bargap=0.03)
    st.plotly_chart(fig_hist, use_container_width=True, theme=None)

# ──────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️  About this simulation", expanded=False):
    st.markdown("""
**Model:** Geometric Brownian Motion (GBM)

**Monte Carlo method:**
- Simulate N stock-price paths over 252 daily steps
- Payoff at expiration: max(S_T - K, 0)
- Discounted price: average of all payoffs times e^(-rT)

**Black-Scholes closed-form** is used as benchmark.
""")
    st.table({
        "Parameter": ["Stock Price (S)", "Strike Price (K)", "Risk-Free Rate (r)", "Volatility (σ)", "Time (T)", "Simulations (N)"],
        "Value": [S, K, r, sigma, T, int(N)]
    })