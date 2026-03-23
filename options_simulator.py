# Monte Carlo Options Pricing Simulator
# Streamlit + Plotly Edition

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Options Pricer",
    layout="wide"
)

# Sidebar inputs
with st.sidebar:
    st.markdown("### Parameters")
    st.markdown("#### Market Data")

    S = st.number_input("Stock Price (S)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=10000.0, value=105.0, step=1.0, format="%.2f")
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=5.0, value=0.20, step=0.01, format="%.4f")

    st.markdown("#### Simulation")

    T = st.number_input("Time to Expiration (years)", min_value=0.01, max_value=10.0, value=1.0, step=0.25, format="%.2f")
    N = st.number_input("Simulations (N)", min_value=100, max_value=500000, value=10000, step=1000)

    st.markdown("---")
    st.markdown("Reruns automatically on input change.")


# Core math — Black-Scholes
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Core math — Monte Carlo GBM
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


# Run simulation
mc_price, std_error, price_paths, payoffs = monte_carlo_call(S, K, T, r, sigma, int(N))
bs_price = black_scholes_call(S, K, T, r, sigma)
difference = mc_price - bs_price


# Header
st.markdown("# Monte Carlo Options Pricer")
st.markdown("European Call Option · Geometric Brownian Motion · Black-Scholes Benchmark")
st.markdown("---")


# Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Monte Carlo Price", value=f"${mc_price:,.4f}")
with col2:
    st.metric(label="Black-Scholes Price", value=f"${bs_price:,.4f}")
with col3:
    st.metric(label="Difference (MC - BS)", value=f"${difference:,.4f}", delta=f"{difference:+.4f}", delta_color="normal")
with col4:
    st.metric(label="MC Std Error", value=f"${std_error:,.4f}")

st.markdown("---")


# Plotly theming
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Playfair Display", color="#111111", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(
        gridcolor="#eeeeee",
        linecolor="#111111",
        linewidth=1.5,
        tickfont=dict(size=10, color="#555555"),
        title_font=dict(size=10, color="#555555"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#eeeeee",
        linecolor="#111111",
        linewidth=1.5,
        tickfont=dict(size=10, color="#555555"),
        title_font=dict(size=10, color="#555555"),
        zeroline=False,
    ),
)


# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Simulated Price Paths (first 100)")

    n_display = min(100, int(N))
    t_axis = np.linspace(0, T, price_paths.shape[1])
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
        annotation_text=f"Strike K = {K}",
        annotation_position="top left",
        annotation_font=dict(size=10),
    )
    fig_paths.add_hline(
        y=S, line_dash="dot", line_color="#111111", line_width=1.2,
        annotation_text=f"S0 = {S}",
        annotation_position="bottom left",
        annotation_font=dict(size=10),
    )
    fig_paths.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Time (years)", yaxis_title="Stock Price ($)")
    st.plotly_chart(fig_paths, use_container_width=True, theme=None)

with chart_col2:
    st.markdown("#### Payoff Distribution at Expiration")

    mean_payoff = np.mean(payoffs)
    itm_pct = np.mean(payoffs > 0) * 100
    fig_hist = go.Figure()

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
        annotation_text=f"Mean ${mean_payoff:,.2f}",
        annotation_position="top right",
        annotation_font=dict(size=10),
    )
    fig_hist.add_annotation(
        x=0.97, y=0.94, xref="paper", yref="paper",
        text=f"ITM: {itm_pct:.1f}%    OTM: {100 - itm_pct:.1f}%",
        showarrow=False,
        font=dict(size=10),
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=6,
        xanchor="right",
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Call Payoff ($)", yaxis_title="Frequency", bargap=0.02)
    st.plotly_chart(fig_hist, use_container_width=True, theme=None)


# Footer
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
        "Value": [S, K, r, sigma, T, int(N)]
    })
