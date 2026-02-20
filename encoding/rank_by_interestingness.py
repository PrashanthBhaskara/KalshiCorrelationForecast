"""
Rank market pairs by "interestingness" for research purposes.
Balances similarity with semantic diversity.
"""

import pandas as pd
import numpy as np
import re

def calculate_interestingness_score(row):
    """
    Calculate an interestingness score for a market pair.

    Factors:
    1. Cross-category bonus (different categories = more interesting)
    2. Similarity score (need some correlation, but not perfect)
    3. Semantic diversity penalty (same asset/topic = less interesting)
    """
    similarity = row['similarity']

    # Base score from similarity
    # Optimal range: 0.75-0.90 (correlated but not identical)
    if similarity > 0.95:
        # Too similar - probably same phenomenon
        similarity_score = 0.5
    elif 0.85 <= similarity <= 0.95:
        # Sweet spot - correlated but distinct
        similarity_score = 1.0
    elif 0.75 <= similarity < 0.85:
        # Still good
        similarity_score = 0.9
    else:
        # Lower correlation, less useful
        similarity_score = similarity * 0.8

    # Cross-category bonus
    if row['cross_category']:
        category_bonus = 1.5  # 50% boost for cross-category
    else:
        category_bonus = 1.0

    # Semantic diversity: penalize Bitcoin vs Bitcoin, CPI vs CPI, etc.
    diversity_penalty = calculate_semantic_diversity(row)

    # Volume factor (higher volume = more liquid/reliable)
    min_volume = min(row['volume1'], row['volume2'])
    if min_volume > 5_000_000:
        volume_factor = 1.2
    elif min_volume > 2_000_000:
        volume_factor = 1.1
    else:
        volume_factor = 1.0

    # Final score
    score = similarity_score * category_bonus * diversity_penalty * volume_factor

    return score

def calculate_semantic_diversity(row):
    """
    Penalize pairs that are semantically too similar.
    """
    t1 = row['title1'].lower()
    t2 = row['title2'].lower()

    # Extract main subject/asset
    def extract_subject(text):
        # Bitcoin patterns
        if 'bitcoin' in text or 'btc' in text:
            return 'bitcoin'
        # CPI/Inflation
        if 'cpi' in text or 'inflation' in text:
            return 'cpi'
        # Grammy/Awards
        if 'grammy' in text or 'emmy' in text or 'award' in text:
            return 'awards'
        # Fed/Interest rates
        if 'fed' in text or 'federal reserve' in text or 'interest rate' in text:
            return 'fed'
        # Elections
        if 'election' in text or 'vote' in text:
            return 'election'
        # Generic
        return 'other'

    subject1 = extract_subject(t1)
    subject2 = extract_subject(t2)

    # Specific penalties
    # Bitcoin price vs Bitcoin price range = very low diversity
    if subject1 == 'bitcoin' and subject2 == 'bitcoin':
        if ('price range' in t1 and 'price on' in t2) or ('price range' in t2 and 'price on' in t1):
            return 0.3  # Heavy penalty
        return 0.6  # Moderate penalty for Bitcoin vs Bitcoin

    # Same subject = penalty
    if subject1 == subject2:
        return 0.7

    # Different subjects = good
    return 1.0

def rank_by_interestingness(input_csv: str, output_csv: str, top_n: int = 100):
    """
    Re-rank pairs by interestingness score.
    """
    df = pd.read_csv(input_csv)

    print(f"Loaded {len(df)} pairs")

    # Calculate interestingness score
    df['interestingness_score'] = df.apply(calculate_interestingness_score, axis=1)

    # Sort by interestingness
    df_ranked = df.sort_values('interestingness_score', ascending=False)

    # Save full ranked list
    df_ranked.to_csv(output_csv, index=False)

    # Also save top N
    top_n_path = output_csv.replace('.csv', f'_top{top_n}.csv')
    df_ranked.head(top_n).to_csv(top_n_path, index=False)

    print(f"\nSaved full ranked list to: {output_csv}")
    print(f"Saved top {top_n} to: {top_n_path}")

    return df_ranked

def print_top_interesting(df: pd.DataFrame, top_n: int = 50):
    """Print top interesting pairs."""
    print(f"\n{'='*100}")
    print(f"TOP {top_n} MOST INTERESTING MARKET PAIRS")
    print(f"(Ranked by interestingness score)")
    print(f"{'='*100}")

    for idx, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        cross_cat = "🔄 CROSS-CAT" if row['cross_category'] else "📊 SAME-CAT"

        print(f"\n{idx}. Score: {row['interestingness_score']:.3f} | Similarity: {row['similarity']:.4f} {cross_cat}")
        print(f"   [{row['ticker1']}] {row['category1']}")
        print(f"   \"{row['title1'][:80]}\"")
        print(f"   Vol: ${row['volume1']:,.0f}")
        print(f"   ↔")
        print(f"   [{row['ticker2']}] {row['category2']}")
        print(f"   \"{row['title2'][:80]}\"")
        print(f"   Vol: ${row['volume2']:,.0f}")
        print(f"   " + "-"*96)

def analyze_top_patterns(df: pd.DataFrame, top_n: int = 100):
    """Analyze patterns in top N pairs."""
    top = df.head(top_n)

    print(f"\n{'='*100}")
    print(f"ANALYSIS OF TOP {top_n} PAIRS")
    print(f"{'='*100}")

    print(f"\nCross-category breakdown:")
    print(f"  Cross-category: {top['cross_category'].sum()} pairs ({100*top['cross_category'].sum()/len(top):.1f}%)")
    print(f"  Same-category: {(~top['cross_category']).sum()} pairs ({100*(~top['cross_category']).sum()/len(top):.1f}%)")

    print(f"\nCategory pair distribution:")
    top['category_pair'] = top.apply(
        lambda row: f"{row['category1']} ↔ {row['category2']}" if row['category1'] <= row['category2']
                    else f"{row['category2']} ↔ {row['category1']}",
        axis=1
    )

    for pair, count in top['category_pair'].value_counts().head(10).items():
        avg_sim = top[top['category_pair'] == pair]['similarity'].mean()
        avg_score = top[top['category_pair'] == pair]['interestingness_score'].mean()
        print(f"  {pair}: {count} pairs (avg sim: {avg_sim:.3f}, avg score: {avg_score:.3f})")

    print(f"\nSimilarity range:")
    print(f"  Min: {top['similarity'].min():.4f}")
    print(f"  Max: {top['similarity'].max():.4f}")
    print(f"  Mean: {top['similarity'].mean():.4f}")
    print(f"  Median: {top['similarity'].median():.4f}")

def main():
    # Rank by interestingness
    df_ranked = rank_by_interestingness(
        'encoding/interesting_correlations.csv',
        'encoding/ranked_interesting_correlations.csv',
        top_n=100
    )

    # Print top results
    print_top_interesting(df_ranked, top_n=50)

    # Analyze patterns
    analyze_top_patterns(df_ranked, top_n=100)

    print(f"\n{'='*100}")
    print("DONE! Check ranked_interesting_correlations.csv and _top100.csv")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()