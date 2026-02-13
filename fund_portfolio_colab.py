"""Colab-friendly script to build an investment fund portfolio.

Features:
- Reads 3 CSV files: performance, management fees, volume changes.
- Prioritizes recent volume growth.
- Treats management fee as low-priority.
- Applies user criteria: risk level, fund overlap, portfolio time window.
- Uses Groq LLM for final qualitative scoring and short rationale.
- Uses Serper web search to check market commentary and buy-vs-sell leaning.

Usage (in Colab cell):

from fund_portfolio_colab import (
    PortfolioCriteria,
    build_validated_portfolio,
)

criteria = PortfolioCriteria(
    risk_level="medium",
    max_fund_type_overlap=0.5,
    time_window="1y",
    top_n=8,
)

result = build_validated_portfolio(
    performance_csv="performance.csv",
    fees_csv="fees.csv",
    volume_csv="volume.csv",
    groq_api_key="<YOUR_GROQ_KEY>",
    serper_api_key="<YOUR_SERPER_KEY>",
    criteria=criteria,
)

result[["fund_acronym", "fund_name", "weight_pct", "selection_reason"]]
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class PortfolioCriteria:
    """User criteria for portfolio construction."""

    risk_level: str = "medium"  # low, medium, high
    max_fund_type_overlap: float = 0.6  # 0..1, max portfolio share from same type
    time_window: str = "1y"  # e.g. 3m, 6m, 1y, 3y
    top_n: int = 8
    min_positive_sentiment_share: float = 0.6


def _first_matching_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise KeyError(f"None of candidate columns found: {candidates}. Found: {list(df.columns)}")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


def _resolve_columns(perf: pd.DataFrame, fees: pd.DataFrame, vol: pd.DataFrame, time_window: str) -> Dict[str, str]:
    perf_return_candidates = [
        f"return_{time_window}",
        f"perf_{time_window}",
        f"performance_{time_window}",
        "return_1y",
        "annual_return",
        "performance",
    ]
    return {
        "key": _first_matching_column(perf, ["fund_acronym", "ticker", "symbol", "fund_id"]),
        "name": _first_matching_column(perf, ["fund_name", "name"]),
        "fund_type": _first_matching_column(perf, ["fund_type", "category", "asset_class"]),
        "risk": _first_matching_column(perf, ["risk_score", "risk", "volatility"]),
        "performance": _first_matching_column(perf, perf_return_candidates),
        "fee": _first_matching_column(fees, ["management_fee", "fee_pct", "expense_ratio", "fee"]),
        "volume_change": _first_matching_column(vol, ["volume_change_recent", "recent_volume_change", "aum_change", "volume_change"]),
    }


def _merge_data(performance_csv: str, fees_csv: str, volume_csv: str, time_window: str) -> pd.DataFrame:
    perf = pd.read_csv(performance_csv)
    fees = pd.read_csv(fees_csv)
    vol = pd.read_csv(volume_csv)

    cols = _resolve_columns(perf, fees, vol, time_window)

    key_perf = cols["key"]
    key_fees = _first_matching_column(fees, ["fund_acronym", "ticker", "symbol", "fund_id", key_perf])
    key_vol = _first_matching_column(vol, ["fund_acronym", "ticker", "symbol", "fund_id", key_perf])

    merged = (
        perf[[key_perf, cols["name"], cols["fund_type"], cols["risk"], cols["performance"]]]
        .merge(fees[[key_fees, cols["fee"]]], left_on=key_perf, right_on=key_fees, how="inner")
        .merge(vol[[key_vol, cols["volume_change"]]], left_on=key_perf, right_on=key_vol, how="inner")
    )

    merged = merged.rename(
        columns={
            key_perf: "fund_acronym",
            cols["name"]: "fund_name",
            cols["fund_type"]: "fund_type",
            cols["risk"]: "risk_score",
            cols["performance"]: "performance_metric",
            cols["fee"]: "management_fee",
            cols["volume_change"]: "recent_volume_change",
        }
    )

    for c in ["risk_score", "performance_metric", "management_fee", "recent_volume_change"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged = merged.dropna(subset=["fund_acronym", "fund_name", "fund_type", "risk_score", "performance_metric", "management_fee", "recent_volume_change"])
    return merged


def _score_funds(df: pd.DataFrame, criteria: PortfolioCriteria) -> pd.DataFrame:
    risk_targets = {"low": 0.3, "medium": 0.6, "high": 0.85}
    target = risk_targets.get(criteria.risk_level.lower(), 0.6)

    scored = df.copy()
    scored["volume_score"] = _zscore(scored["recent_volume_change"])
    scored["performance_score"] = _zscore(scored["performance_metric"])
    scored["fee_score"] = -_zscore(scored["management_fee"])  # lower fee is better, but low priority
    scored["risk_fit_score"] = -abs(scored["risk_score"] - target)
    scored["risk_fit_score"] = _zscore(scored["risk_fit_score"])

    scored["composite_score"] = (
        0.50 * scored["volume_score"]
        + 0.35 * scored["performance_score"]
        + 0.10 * scored["risk_fit_score"]
        + 0.05 * scored["fee_score"]
    )

    return scored.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _enforce_overlap_limit(scored: pd.DataFrame, criteria: PortfolioCriteria) -> pd.DataFrame:
    selected_rows: List[pd.Series] = []
    per_type_count: Dict[str, int] = {}
    max_per_type = max(1, int(np.floor(criteria.max_fund_type_overlap * criteria.top_n)))

    for _, row in scored.iterrows():
        ftype = str(row["fund_type"])
        if per_type_count.get(ftype, 0) >= max_per_type:
            continue
        selected_rows.append(row)
        per_type_count[ftype] = per_type_count.get(ftype, 0) + 1
        if len(selected_rows) >= criteria.top_n:
            break

    if not selected_rows:
        return scored.head(criteria.top_n).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _serper_search(serper_api_key: str, query: str, k: int = 5) -> List[str]:
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
        json={"q": query, "num": k},
        timeout=25,
    )
    response.raise_for_status()
    data = response.json()
    snippets = []
    for item in data.get("organic", [])[:k]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        snippets.append(f"{title}: {snippet}".strip())
    return snippets


def _groq_chat(groq_api_key: str, model: str, prompt: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _web_sentiment_check(groq_api_key: str, serper_api_key: str, fund_acronym: str, fund_name: str, model: str) -> Tuple[str, float]:
    query = f"{fund_acronym} {fund_name} fund analyst opinion buy sell"
    snippets = _serper_search(serper_api_key, query, k=6)
    if not snippets:
        return "neutral", 0.0

    prompt = (
        "Given these web snippets about an investment fund, classify overall commentary and buy-vs-sell leaning.\n"
        "Return only JSON with keys: sentiment (positive|neutral|negative), buy_score (-1 to 1), explanation (max 20 words).\n\n"
        f"Fund: {fund_acronym} - {fund_name}\n"
        f"Snippets:\n- " + "\n- ".join(snippets)
    )

    raw = _groq_chat(groq_api_key, model, prompt)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return "neutral", 0.0
    try:
        obj = json.loads(raw[start : end + 1])
        sentiment = str(obj.get("sentiment", "neutral")).lower().strip()
        buy_score = float(obj.get("buy_score", 0.0))
        return sentiment, buy_score
    except (json.JSONDecodeError, ValueError, TypeError):
        return "neutral", 0.0


def _llm_reason(groq_api_key: str, model: str, row: pd.Series) -> str:
    prompt = (
        "Write one sentence (max 22 words) why this fund is selected. "
        "Prioritize recent volume growth, then performance, then risk fit. Fee is low priority.\n"
        f"Fund: {row['fund_acronym']} - {row['fund_name']}\n"
        f"Recent volume change: {row['recent_volume_change']}\n"
        f"Performance metric: {row['performance_metric']}\n"
        f"Risk score: {row['risk_score']}\n"
        f"Management fee: {row['management_fee']}\n"
        f"Web sentiment: {row['web_sentiment']}\n"
        f"Buy score: {row['web_buy_score']}\n"
    )
    return _groq_chat(groq_api_key, model, prompt).strip().replace("\n", " ")


def build_validated_portfolio(
    performance_csv: str,
    fees_csv: str,
    volume_csv: str,
    groq_api_key: str,
    serper_api_key: str,
    criteria: PortfolioCriteria,
    groq_model: str = "llama-3.3-70b-versatile",
) -> pd.DataFrame:
    """Build portfolio and validate picks with web sentiment checks."""
    merged = _merge_data(performance_csv, fees_csv, volume_csv, criteria.time_window)
    scored = _score_funds(merged, criteria)
    selected = _enforce_overlap_limit(scored, criteria)

    sentiments = []
    for _, row in selected.iterrows():
        sentiment, buy_score = _web_sentiment_check(
            groq_api_key,
            serper_api_key,
            row["fund_acronym"],
            row["fund_name"],
            groq_model,
        )
        sentiments.append((sentiment, buy_score))

    selected = selected.copy()
    selected["web_sentiment"] = [s[0] for s in sentiments]
    selected["web_buy_score"] = [s[1] for s in sentiments]

    filtered = selected[
        (selected["web_sentiment"] != "negative")
        & (selected["web_buy_score"] >= 0)
    ].copy()

    if filtered.empty:
        filtered = selected.nlargest(max(1, criteria.top_n // 2), "composite_score").copy()

    weights = _softmax(filtered["composite_score"].to_numpy())
    filtered["weight_pct"] = np.round(weights * 100, 2)
    filtered["weight_pct"] = np.round(filtered["weight_pct"] / filtered["weight_pct"].sum() * 100, 2)

    # fix potential rounding residual on top fund
    residual = 100.0 - filtered["weight_pct"].sum()
    if abs(residual) >= 0.01:
        top_idx = filtered["weight_pct"].idxmax()
        filtered.loc[top_idx, "weight_pct"] = np.round(filtered.loc[top_idx, "weight_pct"] + residual, 2)

    reasons = []
    for _, row in filtered.iterrows():
        reasons.append(_llm_reason(groq_api_key, groq_model, row))
    filtered["selection_reason"] = reasons

    cols = [
        "fund_acronym",
        "fund_name",
        "fund_type",
        "weight_pct",
        "selection_reason",
        "recent_volume_change",
        "performance_metric",
        "management_fee",
        "web_sentiment",
        "web_buy_score",
    ]
    return filtered[cols].sort_values("weight_pct", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Minimal CLI example for local usage.
    # In Colab, prefer importing build_validated_portfolio directly.
    import argparse

    parser = argparse.ArgumentParser(description="Build a validated investment fund portfolio.")
    parser.add_argument("--performance_csv", required=True)
    parser.add_argument("--fees_csv", required=True)
    parser.add_argument("--volume_csv", required=True)
    parser.add_argument("--groq_api_key", required=True)
    parser.add_argument("--serper_api_key", required=True)
    parser.add_argument("--risk_level", default="medium")
    parser.add_argument("--max_overlap", type=float, default=0.6)
    parser.add_argument("--time_window", default="1y")
    parser.add_argument("--top_n", type=int, default=8)

    args = parser.parse_args()

    criteria = PortfolioCriteria(
        risk_level=args.risk_level,
        max_fund_type_overlap=args.max_overlap,
        time_window=args.time_window,
        top_n=args.top_n,
    )

    out = build_validated_portfolio(
        performance_csv=args.performance_csv,
        fees_csv=args.fees_csv,
        volume_csv=args.volume_csv,
        groq_api_key=args.groq_api_key,
        serper_api_key=args.serper_api_key,
        criteria=criteria,
    )
    print(out.to_string(index=False))
