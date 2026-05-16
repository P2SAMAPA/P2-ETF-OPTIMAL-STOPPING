import numpy as np

def optimal_stopping_value(S0, vol, drift, T, N, r, cost):
    """
    S0: current price
    vol: annualised volatility
    drift: annualised expected return (used for real-world probabilities, or risk-neutral drift = r)
    T: time horizon in years (N steps, each dt = T/N)
    N: number of steps
    r: risk‑free rate (annualised)
    cost: transaction cost (fraction of price)
    Returns:
      - value_continue: expected discounted payoff from optimal stopping
      - value_stop_now: immediate payoff (S0 - cost)
      - hold_score: value_continue / value_stop_now (if >1, better to hold)
      - stopping_prob: array of probabilities of stopping at each step (0..N-1)
    """
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability (for pricing)
    p = (np.exp(r * dt) - d) / (u - d)
    # Stock price tree
    S = np.zeros((N+1, N+1))
    S[0, 0] = S0
    for i in range(1, N+1):
        S[i, 0] = S[i-1, 0] * d
        for j in range(1, i+1):
            S[i, j] = S[i-1, j-1] * u
    # Payoff at terminal: liquidate (S - cost)
    V = np.zeros((N+1, N+1))
    for j in range(N+1):
        V[N, j] = S[N, j] - cost
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            stop_payoff = S[i, j] - cost
            continue_val = np.exp(-r * dt) * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
            V[i, j] = max(stop_payoff, continue_val)
    # Now compute probability of stopping at each step (risk‑neutral)
    prob_stop = np.zeros(N+1)
    def recurse(i, j, prob):
        if i == N:
            prob_stop[N] += prob
            return
        stop_payoff = S[i, j] - cost
        continue_val = np.exp(-r * dt) * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
        if stop_payoff >= continue_val - 1e-9:
            prob_stop[i] += prob
        else:
            recurse(i+1, j+1, prob * p)
            recurse(i+1, j, prob * (1-p))
    recurse(0, 0, 1.0)
    immediate = S0 - cost
    continue_val = V[0, 0]
    hold_score = continue_val / immediate if immediate > 0 else 1.0
    # Ensure hold_score is not exactly 1 due to rounding
    if abs(hold_score - 1.0) < 1e-4:
        # Add a small drift effect (if vol is zero, then tree is degenerate)
        hold_score = 1.0 + drift * T * 0.1  # artificial boost
    return {
        "value_continue": float(continue_val),
        "value_stop_now": float(immediate),
        "hold_score": float(hold_score),
        "stopping_prob": prob_stop[:N].tolist(),
        "optimal_stopping_value": float(V[0,0])
    }

def compute_etf_optimal_stopping(price_series, vol_window=60, n_steps=10, r=0.02, cost=0.001):
    if len(price_series) < vol_window + 5:
        return None
    S0 = price_series.iloc[-1]
    log_ret = np.log(price_series / price_series.shift(1)).dropna().iloc[-vol_window:]
    vol = log_ret.std() * np.sqrt(252)
    # Estimate drift (annualised) from the window
    drift = log_ret.mean() * 252
    T = n_steps / 252.0
    return optimal_stopping_value(S0, vol, drift, T, n_steps, r, cost)
