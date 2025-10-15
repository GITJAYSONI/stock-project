# stock-AI-AGENT

Simple Streamlit-based stock monitor plus helper scripts to prepare data for LSTM training.

What's included
- `data_extraction.py` — Streamlit app to fetch and view intraday stock data using yfinance.
- `prepare_data_for_lstm.py` — Script to fetch historical data, scale closes, and save training sequences.
- `data/` — (generated) contains raw CSVs and .npy datasets for training.
- `models/` — (generated) contains saved scalers.

Quick start (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
streamlit run "c:\\Users\\jayso\\OneDrive\\Desktop\\New folder\\data_extraction.py"
```

Notes
- I added a `.gitignore` to exclude `venv/`, `data/`, and `models/` from future commits. The repository currently contains these files; you may want to remove the `venv` directory from the repo history to reduce size.
- If you want, I can remove `venv` from the repository history and force-push a cleaned branch — say the word and I'll prepare the steps.

Enjoy!
# stock-AI-AGENT