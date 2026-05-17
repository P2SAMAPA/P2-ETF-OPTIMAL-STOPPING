import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from optimal_stopping import compute_etf_optimal_stopping

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Optimal Stopping) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        if prices.empty or len(prices) < max(config.WINDOWS) + config.VOL_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(prices) < win + config.VOL_WINDOW + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_scores = {}
            etf_details = {}
            for etf in tickers:
                if etf not in prices.columns:
                    continue
                # Use last `win` days of price data to estimate volatility and compute optimal stopping
                price_series = prices[etf].iloc[-win:].dropna()
                if len(price_series) < config.VOL_WINDOW + 5:
                    continue
                res = compute_etf_optimal_stopping(
                    price_series,
                    vol_window=config.VOL_WINDOW,
                    n_steps=config.N_STEPS,
                    r=config.RISK_FREE_RATE,
                    cost=config.TRANSACTION_COST
                )
                if res is None:
                    continue
                hold_score = res["hold_score"]
                etf_scores[etf] = hold_score
                etf_details[etf] = res
            window_results[win] = {"scores": etf_scores, "details": etf_details}
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            # Fallback: use historical mean return (positive if >0, else small positive)
            print("  No valid predictions – falling back to historical mean return")
            returns = data_manager.prepare_returns_matrix(df, tickers)
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)  # ensure non‑zero
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "hold_score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/optimal_stopping_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Optimal Stopping Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
