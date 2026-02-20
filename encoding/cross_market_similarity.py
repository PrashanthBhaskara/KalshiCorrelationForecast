"""
Cross-Market Similarity Analysis
Filters out same-series time variants to find truly interesting correlations.
"""

import pandas as pd
import numpy as np
import re

def extract_series_prefix(ticker: str) -> str:
    """
    Extract the base series from a ticker by removing date/time suffixes.
    Examples:
        KXBTCD-26JAN2717 -> KXBTCD
        KXCPI-25APR -> KXCPI
        KXFEDDECISION-25SEP -> KXFEDDECISION
    """
    # Remove common date patterns
    # Pattern: hyphen followed by year/month/day/time info
    base = re.sub(r'-\d{2}[A-Z]{3}\d{0,4}$', '', ticker)  # -26JAN2717, -25APR
    base = re.sub(r'-\d{4}$', '', base)  # -2024, -2025
    base = re.sub(r'-[A-Z]{3}\d{2}$', '', base)  # -JAN26

    return base

def are_same_series(ticker1: str, ticker2: str) -> bool:
    """Check if two tickers are from the same market series."""
    series1 = extract_series_prefix(ticker1)
    series2 = extract_series_prefix(ticker2)

    return series1 == series2

def titles_differ_only_by_dates(title1: str, title2: str) -> bool:
    """
    Check if two titles are essentially the same except for dates.
    """
    # Remove all date-like patterns
    def remove_dates(text):
        # Remove years (2024, 2025, etc.)
        text = re.sub(r'\b20\d{2}\b', 'YEAR', text)
        # Remove month names
        text = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', 'MONTH', text, flags=re.IGNORECASE)
        # Remove day numbers (1st, 2nd, 21, 22, etc.)
        text = re.sub(r'\b\d{1,2}(st|nd|rd|th)?\b', 'DAY', text)
        # Remove "at Xpm" or "at X:XXpm"
        text = re.sub(r'at \d{1,2}(:\d{2})?\s*(am|pm|AM|PM|EST|PST)', 'at TIME', text)
        # Remove "Before/After DATE"
        text = re.sub(r'\b(Before|After|In|On)\s+[A-Z]', 'WHEN', text)

        return text.strip()

    normalized1 = remove_dates(title1)
    normalized2 = remove_dates(title2)

    # If they're identical after removing dates, they're time variants
    return normalized1 == normalized2

def filter_cross_market_pairs(input_csv: str, output_csv: str, min_similarity: float = 0.80):
    """
    Filter to find cross-market similarities (not same-series time variants).
    """
    df = pd.read_csv(input_csv)

    print(f"Starting with {len(df)} pairs with similarity > {min_similarity}")

    # Filter 1: Remove same-series pairs
    df['same_series'] = df.apply(lambda row: are_same_series(row['ticker1'], row['ticker2']), axis=1)
    cross_series = df[~df['same_series']].copy()

    print(f"After removing same-series pairs: {len(cross_series)} pairs")

    # Filter 2: Remove titles that differ only by dates
    cross_series['date_variants'] = cross_series.apply(
        lambda row: titles_differ_only_by_dates(row['title1'], row['title2']),
        axis=1
    )
    truly_different = cross_series[~cross_series['date_variants']].copy()

    print(f"After removing date-only variants: {len(truly_different)} pairs")

    # Sort by similarity descending
    truly_different = truly_different.sort_values('similarity', ascending=False)

    # Save
    truly_different.to_csv(output_csv, index=False)

    return truly_different

def print_cross_market_pairs(df: pd.DataFrame, top_n: int = 30):
    """Pretty print cross-market pairs."""
    print(f"\n{'='*100}")
    print(f"TOP {top_n} CROSS-MARKET SIMILARITIES")
    print(f"(Filtered to exclude same-series time variants)")
    print(f"{'='*100}")

    for idx, row in df.head(top_n).iterrows():
        print(f"\n{idx+1}. Similarity: {row['similarity']:.4f}")
        print(f"   [{row['ticker1']}] {row['category1']}")
        print(f"   \"{row['title1']}\"")
        print(f"   Volume: ${row['volume1']:,.0f}")
        print(f"   ---")
        print(f"   [{row['ticker2']}] {row['category2']}")
        print(f"   \"{row['title2']}\"")
        print(f"   Volume: ${row['volume2']:,.0f}")
        print(f"   " + "-"*96)

def analyze_category_patterns(df: pd.DataFrame):
    """Analyze which category pairs are most common."""
    print(f"\n{'='*100}")
    print("CATEGORY PAIR PATTERNS")
    print(f"{'='*100}")

    # Create category pair labels
    df['category_pair'] = df.apply(
        lambda row: tuple(sorted([row['category1'], row['category2']])),
        axis=1
    )

    category_counts = df['category_pair'].value_counts().head(15)

    for (cat1, cat2), count in category_counts.items():
        if cat1 == cat2:
            print(f"  {cat1} ↔ {cat2}: {count} pairs (same category)")
        else:
            print(f"  {cat1} ↔ {cat2}: {count} pairs")

def main():
    # Filter the high similarity pairs
    cross_market_df = filter_cross_market_pairs(
        'encoding/high_similarity_pairs_above_0.80.csv',
        'encoding/cross_market_similarities.csv',
        min_similarity=0.80
    )

    # Print top results
    print_cross_market_pairs(cross_market_df, top_n=30)

    # Analyze patterns
    analyze_category_patterns(cross_market_df)

    print(f"\n{'='*100}")
    print(f"Saved {len(cross_market_df)} cross-market pairs to encoding/cross_market_similarities.csv")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()