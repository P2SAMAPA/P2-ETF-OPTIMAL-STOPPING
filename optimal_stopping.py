import numpy as np

def optimal_stopping_value(S0, vol, T, N, r, cost):
    """
    Compute optimal stopping value for a single ETF.
    S0: current price
    vol: annualised volatility
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
    p = (np.exp(r * dt) - d) / (u - d)   # risk‑neutral probability
    # Terminal payoffs at step N: stock price at each node (from binomial)
    # We compute stock price tree
    S = np.zeros((N+1, N+1))
    S[0, 0] = S0
    for i in range(1, N+1):
        S[i, 0] = S[i-1, 0] * d
        for j in range(1, i+1):
            S[i, j] = S[i-1, j-1] * u
    # Payoff at terminal (step N): immediate liquidation (S - cost)
    V = np.zeros((N+1, N+1))
    for j in range(N+1):
        V[N, j] = S[N, j] - cost   # can exit at final step
    # Stopping probability: we will record the probability that optimal decision is to stop at each step
    stop_prob = np.zeros(N+1)   # index = step number
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            # Immediate payoff at this node
            stop_payoff = S[i, j] - cost
            # Expected continuation value
            continue_value = np.exp(-r * dt) * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
            # Optimal decision
            V[i, j] = max(stop_payoff, continue_value)
            # If stop_payoff > continue_value, optimal to stop here
            # We will accumulate probability of stopping at this node (under risk‑neutral measure)
            # Need probability of reaching this node: binomial probability
            # We'll compute later: for initial node (0,0), we only care about probability mass where stop is optimal
            # For simplicity, we compute probability that optimal stopping occurs at each step using the tree.
    # Now compute probability of stopping at each step (starting from initial)
    # We'll simulate with risk‑neutral probabilities.
    prob_stop = np.zeros(N+1)
    # Use recursion
    def recurse(i, j, prob):
        if i == N:
            # At terminal, we stop (already counted in prob_stop[N])
            prob_stop[N] += prob
            return
        stop_payoff = S[i, j] - cost
        continue_val = np.exp(-r * dt) * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
        if stop_payoff >= continue_val - 1e-9:   # optimal to stop
            prob_stop[i] += prob
        else:
            # continue: split probability to up and down
            recurse(i+1, j+1, prob * p)
            recurse(i+1, j, prob * (1-p))
    recurse(0, 0, 1.0)
    # Expected gain from waiting vs immediate exit
    immediate = S0 - cost
    continue_val = V[0, 0]
    hold_score = continue_val / immediate if immediate > 0 else 1.0
    return {
        "value_continue": float(continue_val),
        "value_stop_now": float(immediate),
        "hold_score": float(hold_score),
        "stopping_prob": prob_stop[:N].tolist(),   # up to N-1 (plus terminal? we'll exclude terminal)
        "optimal_stopping_value": float(V[0,0])
    }

def compute_etf_optimal_stopping(price_series, vol_window=60, n_steps=10, r=0.02, cost=0.001):
    """
    price_series: pandas Series with datetime index.
    Returns optimal stopping result.
    """
    if len(price_series) < vol_window + 5:
        return None
    # Current price
    S0 = price_series.iloc[-1]
    # Estimate volatility from log returns over the window
    log_ret = np.log(price_series / price_series.shift(1)).dropna().iloc[-vol_window:]
    vol = log_ret.std() * np.sqrt(252)   # annualised
    # Time horizon: say 10 trading days = 10/252 years
    T = n_steps / 252.0
    return optimal_stopping_value(S0, vol, T, n_steps, r, cost)
