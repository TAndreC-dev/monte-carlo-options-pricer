# Monte Carlo Options Pricer

A web-based European call option pricing simulator built with Python and Streamlit. Simulates 10,000+ stock price paths using Geometric Brownian Motion and benchmarks results against the Black-Scholes analytical formula in real time.

**Live app:** [monte-carlo-options-pricer](https://monte-carlo-options-pricer-z3cuys4da6qketarlk7u3r.streamlit.app/#monte-carlo-options-pricer)]([https://your-app-url.streamlit.app](https://monte-carlo-options-pricer-z3cuys4da6qketarlk7u3r.streamlit.app/#monte-carlo-options-pricer))

---

## What it does

- Simulates N stock price paths over 252 daily time steps using GBM
- Calculates the discounted average payoff across all paths to produce a Monte Carlo option price
- Computes the Black-Scholes closed-form price as a benchmark
- Displays the difference and standard error between the two methods
- Updates all outputs automatically when any input parameter is changed

## Charts

- **Simulated Price Paths** — visualizes the first 100 GBM paths with strike and starting price reference lines
- **Payoff Distribution** — histogram of all simulated call payoffs at expiration, with ITM/OTM breakdown

## Parameters

| Parameter | Description | Default |
|:----------|:------------|--------:|
| Stock Price (S) | Current price of the underlying asset | 100 |
| Strike Price (K) | Option strike price | 105 |
| Risk-Free Rate (r) | Annualized risk-free interest rate | 0.05 |
| Volatility (σ) | Annualized volatility of the underlying | 0.20 |
| Time to Expiration (T) | Time to expiration in years | 1.0 |
| Simulations (N) | Number of Monte Carlo paths | 10,000 |

## Tech stack

- Python
- Streamlit
- NumPy
- SciPy
- Plotly

## Run locally

```bash
pip install streamlit numpy scipy plotly
streamlit run options_simulator.py
```
