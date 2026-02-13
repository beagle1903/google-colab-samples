# google-colab-samples

## Fund portfolio builder for Colab

This repository now includes `fund_portfolio_colab.py`, a Colab-friendly Python script that:

- uses **Groq LLM** (API key provided by you),
- loads **3 CSV files** (performance, management fees, volume changes),
- prioritizes **recent volume growth**,
- treats fee as **low-priority**,
- applies portfolio criteria (risk level, overlap, time window),
- validates selected funds with **Serper web search** + Groq sentiment/buy-vs-sell check,
- outputs fund acronym, name, allocation percentage, and one-sentence reason.

See the top-level docstring in `fund_portfolio_colab.py` for usage examples.
