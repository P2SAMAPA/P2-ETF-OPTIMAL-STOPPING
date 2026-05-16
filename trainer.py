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
        if prices.empty or len(prices) < config.VOL_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        scores = {}
        details = {}
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            price_series = prices[ticker].dropna()
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
            # Score = hold_score (higher = better to hold, lower = better to sell)
            scores[ticker] = res["hold_score"]
            details[ticker] = {
                "hold_score": res["hold_score"],
                "value_continue": res["value_continue"],
                "value_stop_now": res["value_stop_now"],
                "stopping_prob": res["stopping_prob"]
            }

        if not scores:
            print("  No valid results")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Rank by hold_score descending (most attractive to hold)
        sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, score in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "hold_score": float(score)})
            full_scores[ticker] = float(score)
        print(f"  Top 3 ETFs by hold score: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "details": details,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/optimal_stopping_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Optimal Stopping Engine complete ===")

if __name__ == "__main__":
    main()
