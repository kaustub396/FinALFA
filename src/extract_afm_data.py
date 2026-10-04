"""
FinALFA — Feature Extraction & Pre-processing Pipeline
=======================================================
Reads raw company ratio CSVs/XLSXs and filtered-sentiment CSVs,
constructs the sector-weighted Fundamental Score (F_t) and
normalised Sentiment Score (S_t), aligns them with NIFTY-50 annual
returns, and writes the master modelling dataset (final_df.csv) and
fused signal (phase3_fused_signal.csv) to the data/ directory.

Usage
-----
1. Populate the `COMPANY_FILES` list below with your own paths to the
   individual company ratio files.
2. Set `SENTIMENT_CSV` to your filtered news + FinBERT scores CSV.
3. Set `NIFTY_CSV`    to your NIFTY-50 daily OHLCV CSV (Date, Close).
4. Run:  python src/extract_afm_data.py

Output
------
  data/final_df.csv              — master modelling dataset (N=21 obs)
  data/phase3_fused_signal.csv   — per-sector fused signal (before agg)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration — edit these paths before running
# ---------------------------------------------------------------------------

COMPANY_FILES = [
    {"company": "ITC",       "sector": "FMCG",  "path": ""},  # fill in
    {"company": "HUL",       "sector": "FMCG",  "path": ""},
    {"company": "Nestle",    "sector": "FMCG",  "path": ""},
    {"company": "Britannia", "sector": "FMCG",  "path": ""},
    {"company": "Cipla",     "sector": "Pharma", "path": ""},
    {"company": "SunPharma", "sector": "Pharma", "path": ""},
    {"company": "DrReddy",   "sector": "Pharma", "path": ""},
    {"company": "Apollo",    "sector": "Pharma", "path": ""},
]

SENTIMENT_CSV = ""   # path to filtered_fmcg_pharma_news_with_sentiments.csv
NIFTY_CSV     = ""   # path to NIFTY-50 daily OHLCV CSV (columns: Date, Close)
OUT_DIR       = "data"

# Sector weights in NIFTY-50 index (as of dataset construction)
SECTOR_WEIGHTS = {"FMCG": 0.0644, "Pharma": 0.0415}

CANONICAL_COLS = {
    "roe": "ROE", "roe_%": "ROE", "return_on_equity": "ROE",
    "roce": "ROCE", "roce_%": "ROCE", "return_on_capital_employed": "ROCE",
    "current_ratio": "Current_Ratio",
    "debt_equity": "Debt_Equity", "debt_to_equity": "Debt_Equity",
    "pe": "PE", "p_e": "PE",
}

FULL_YEARS = pd.DataFrame({"Year": list(range(2005, 2026))})
RATIO_COLS = ["ROE_mean", "ROCE_mean", "CurrentRatio_mean", "DebtEquity_mean", "PE_mean"]


# ---------------------------------------------------------------------------
# Step 1: Load & normalise company fundamental ratios
# ---------------------------------------------------------------------------

def load_fundamentals(company_files: list) -> pd.DataFrame:
    """Reads per-company ratio files and returns sector-aggregated panel."""
    all_companies = []
    for meta in company_files:
        path = meta["path"]
        if not path:
            raise ValueError(
                f"Path for company '{meta['company']}' is empty. "
                "Please fill in COMPANY_FILES paths before running."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding="latin1")

        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(" ", "_").str.replace("-", "_")
        )
        df = df.rename(columns={k: v for k, v in CANONICAL_COLS.items() if k in df.columns})

        if "year" not in df.columns:
            raise ValueError(f"'year' column missing in {path}")

        df["Year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        keep = ["Year", "ROE", "ROCE", "Current_Ratio", "Debt_Equity", "PE"]
        df = df[[c for c in keep if c in df.columns]]
        df["Company"] = meta["company"]
        df["Sector"] = meta["sector"]
        df = FULL_YEARS.merge(df, on="Year", how="left")
        df["Company"] = meta["company"]
        df["Sector"] = meta["sector"]
        all_companies.append(df)

    master = pd.concat(all_companies, ignore_index=True)
    num_cols = ["ROE", "ROCE", "Current_Ratio", "Debt_Equity", "PE"]
    master[num_cols] = master[num_cols].apply(pd.to_numeric, errors="coerce")

    # Sector-level aggregation
    sector = (
        master.groupby(["Year", "Sector"])
        .agg(
            ROE_mean=("ROE", "mean"),
            ROCE_mean=("ROCE", "mean"),
            CurrentRatio_mean=("Current_Ratio", "mean"),
            DebtEquity_mean=("Debt_Equity", "mean"),
            PE_mean=("PE", "mean"),
            company_count=("Company", "nunique"),
        )
        .reset_index()
    )
    sector[RATIO_COLS] = sector[RATIO_COLS].apply(pd.to_numeric, errors="coerce")

    # Z-score normalise within each sector (prevent look-ahead: fit on all years)
    sector_norm = sector.copy()
    for sec in sector_norm["Sector"].unique():
        mask = sector_norm["Sector"] == sec
        scaler = StandardScaler()
        sector_norm.loc[mask, RATIO_COLS] = scaler.fit_transform(
            sector_norm.loc[mask, RATIO_COLS]
        )

    # Composite fundamental score: positive ratios add, negative subtracted
    sector_norm["Fundamental_Score"] = (
        sector_norm["ROE_mean"]
        + sector_norm["ROCE_mean"]
        + sector_norm["CurrentRatio_mean"]
        - sector_norm["DebtEquity_mean"]
        - sector_norm["PE_mean"]
    )

    return sector_norm[["Year", "Sector", "Fundamental_Score", "company_count"]]


# ---------------------------------------------------------------------------
# Step 2: Load & normalise FinBERT sentiment scores
# ---------------------------------------------------------------------------

def load_sentiment(sentiment_csv: str) -> pd.DataFrame:
    """Aggregates daily weighted FinBERT scores into annual S_t."""
    if not os.path.exists(sentiment_csv):
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_csv}")

    df = pd.read_csv(sentiment_csv)
    df["date"] = pd.to_datetime(df["published_date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df["weighted_sentiment"] = df["sentiment_scores"] * df["Sector"].map(SECTOR_WEIGHTS)

    daily = (
        df.groupby("date")
        .agg(
            index_sentiment=("weighted_sentiment", "sum"),
            total_news=("weighted_sentiment", "count"),
        )
        .reset_index()
    )

    # Fiscal-year mapping: April–March (India)
    daily["FY"] = np.where(
        daily["date"].dt.month >= 4,
        daily["date"].dt.year,
        daily["date"].dt.year - 1,
    )

    annual = (
        daily.groupby("FY")
        .agg(
            Annual_Sentiment=("index_sentiment", "mean"),
            Avg_Daily_News=("total_news", "mean"),
        )
        .reset_index()
        .rename(columns={"FY": "Year"})
    )

    scaler = StandardScaler()
    annual["Annual_Sentiment_Norm"] = scaler.fit_transform(annual[["Annual_Sentiment"]])
    return annual[["Year", "Annual_Sentiment_Norm", "Avg_Daily_News"]]


# ---------------------------------------------------------------------------
# Step 3: Load NIFTY-50 annual log-returns
# ---------------------------------------------------------------------------

def load_nifty(nifty_csv: str) -> pd.DataFrame:
    if not os.path.exists(nifty_csv):
        raise FileNotFoundError(f"NIFTY CSV not found: {nifty_csv}")

    df = pd.read_csv(nifty_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    annual = (
        df.groupby("Year")
        .agg(Close_End=("Close", "last"), Close_Start=("Close", "first"))
        .reset_index()
    )
    annual["annual_return"] = np.log(annual["Close_End"]) - np.log(annual["Close_Start"])
    return annual[["Year", "annual_return"]]


# ---------------------------------------------------------------------------
# Step 4: Fuse and build final modelling dataset
# ---------------------------------------------------------------------------

def build_dataset(fundamentals: pd.DataFrame, sentiment: pd.DataFrame, nifty: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    fusion = fundamentals.merge(sentiment, on="Year", how="inner").sort_values("Year")
    fusion["Fundamental_Score_FF"] = fusion.groupby("Sector")["Fundamental_Score"].ffill()
    fusion["Fusion_Score"] = fusion["Fundamental_Score_FF"] * fusion["Annual_Sentiment_Norm"]

    scaler = StandardScaler()
    fusion["Fusion_Score_Norm"] = scaler.fit_transform(fusion[["Fusion_Score"]])

    phase3 = fusion[
        ["Year", "Sector", "Fundamental_Score_FF", "Annual_Sentiment_Norm", "Fusion_Score_Norm"]
    ].sort_values(["Sector", "Year"]).reset_index(drop=True)
    phase3.to_csv(os.path.join(out_dir, "phase3_fused_signal.csv"), index=False)
    print(f"[Saved] {out_dir}/phase3_fused_signal.csv  — shape {phase3.shape}")

    # Aggregate across sectors for single annual observation
    year_agg = (
        phase3.groupby("Year")
        .agg(
            Fundamental_Score_FF=("Fundamental_Score_FF", "mean"),
            Annual_Sentiment_Norm=("Annual_Sentiment_Norm", "mean"),
            Fusion_Score_Norm=("Fusion_Score_Norm", "mean"),
        )
        .reset_index()
    )

    final = year_agg.merge(nifty, on="Year", how="inner").sort_values("Year").reset_index(drop=True)

    # Forward target: next-year direction (point-in-time, no look-ahead)
    final["market_up_next"] = (final["annual_return"].shift(-1) > 0).astype(int)
    final = final.dropna(subset=["market_up_next"])
    final["market_up_next"] = final["market_up_next"].astype(int)

    final.to_csv(os.path.join(out_dir, "final_df.csv"), index=False)
    print(f"[Saved] {out_dir}/final_df.csv  — shape {final.shape}")
    print(f"Class distribution: {final['market_up_next'].value_counts().to_dict()}")
    print(final[["Year", "Annual_Sentiment_Norm", "Fundamental_Score_FF", "market_up_next"]].to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinALFA feature extraction pipeline")
    parser.add_argument("--sentiment", default=SENTIMENT_CSV, help="Path to sentiment CSV")
    parser.add_argument("--nifty",     default=NIFTY_CSV,     help="Path to NIFTY-50 OHLCV CSV")
    parser.add_argument("--out",       default=OUT_DIR,        help="Output directory for CSVs")
    args = parser.parse_args()

    print("FinALFA — Feature Extraction Pipeline")
    print("=" * 60)

    print("Loading fundamental ratios...")
    fundamentals = load_fundamentals(COMPANY_FILES)

    print("Loading FinBERT sentiment scores...")
    sentiment = load_sentiment(args.sentiment)

    print("Loading NIFTY-50 annual returns...")
    nifty = load_nifty(args.nifty)

    print("Building master dataset...")
    build_dataset(fundamentals, sentiment, nifty, args.out)
    print("Done.")
