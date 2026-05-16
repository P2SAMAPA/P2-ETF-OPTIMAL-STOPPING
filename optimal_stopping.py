import numpy as np

def optimal_stopping_value(S0, vol, T, N, r, cost):
    """
    Compute optimal stopping value for a single ETF.
    """
    dt = T / N
    # Ensure volatility is not zero to avoid degenerate tree
    vol = max(vol, 1e-4)
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    # Risk‑neutral probability (clamp to [0,1] to avoid NaN)
    p_raw = (np.exp(r * dt) - d) / (u - d)
    p = np.clip(p_raw, 0.0, 1.0)
    # Stock price tree
    S = np.zeros((N+1, N+1))
    S[0, 0] = S0
    for i in range(1, N+1):
        S[i, 0] = S[i-1, 0] * d
        for j in range(1, i+1):
            S[i, j] = S[i-1, j-1] * u
    # Payoff at terminal
    V = np.zeros((N+1, N+1))
    for j in range(N+1):
        V[N, j] = S[N, j] - cost
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            stop_payoff = S[i, j] - cost
            continue_val = np.exp(-r * dt) * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
            V[i, j] = max(stop_payoff, continue_val)
    # Compute probability of stopping at each step via recursion
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
    hold_score = V[0, 0] / immediate if immediate > 0 else 1.0
    # If hold_score is exactly 1.0, add tiny differentiation based on volatility
    if abs(hold_score - 1.0) < 1e-6:
        hold_score = 1.0 + 0.01 * vol
    return {
        "value_continue": float(V[0, 0]),
        "value_stop_now": float(immediate),
        "hold_score": float(hold_score),
        "stopping_prob": prob_stop[:N].tolist(),  # up to N-1
        "optimal_stopping_value": float(V[0,0])
    }

def compute_etf_optimal_stopping(price_series, vol_window=60, n_steps=10, r=0.02, cost=0.001):
    """
    price_series: pandas Series with datetime index.
    Returns optimal stopping result.
    """
    if len(price_series) < vol_window + 5:
        return None
    S0 = price_series.iloc[-1]
    log_ret = np.log(price_series / price_series.shift(1)).dropna().iloc[-vol_window:]
    vol = log_ret.std() * np.sqrt(252)
    T = n_steps / 252.0
    return optimal_stopping_value(S0, vol, T, n_steps, r, cost)
