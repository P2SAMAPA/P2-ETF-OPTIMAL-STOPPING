# Optimal Stopping Engine

Solves the optimal stopping problem for ETF exit decisions using a binomial lattice and Snell envelope. Incorporates transaction costs and risk‑neutral probabilities. For each ETF, computes the hold score (value of continuing / value of stopping now). A score >1 suggests waiting is optimal.

- **Lattice:** 10 steps, volatility from 60‑day rolling window
- **Transaction cost:** 0.1% per exit
- **Output:** top 3 ETFs by hold score, plus optimal stopping probability distribution
- **Dashboard:** shows top ETFs, full ranking, and probability histogram

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
