"""
Find Interesting Cross-Market Correlations
Focus on markets from different categories or different fundamental phenomena.
"""

import pandas as pd
import numpy as np

def filter_interesting_correlations(
    input_csv: str,
    output_csv: str,
    min_similarity: float = 0.70,
    prioritize_cross_category: bool = True
):
    """
    Filter for truly interesting correlations:
    1. Cross-category pairs (e.g., Economics ↔ Politics)
    2. Same category but semantically different phenomena
    """
    df = pd.read_csv(input_csv)

    print(f"Starting with {len(df)} cross-market pairs")

    # Separate cross-category and same-category pairs
    df['cross_category'] = df['category1'] != df['category2']

    cross_category = df[df['cross_category']].copy()
    same_category = df[~df['cross_category']].copy()

    print(f"\nCross-category pairs: {len(cross_category)}")
    print(f"Same-category pairs: {len(same_category)}")

    # For same-category, apply stricter filtering
    # Remove pairs where titles are too similar (likely variants)
    def are_title_variants(row):
        """Check if titles are just variants of each other."""
        t1 = row['title1'].lower()
        t2 = row['title2'].lower()

        # Extract key words (remove common words)
        def get_keywords(text):
            words = set(text.split())
            stopwords = {'will', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'a', 'an',
                        'how', 'what', 'who', 'when', 'where', 'is', 'are', 'be', 'and', 'or'}
            return words - stopwords

        kw1 = get_keywords(t1)
        kw2 = get_keywords(t2)

        # Calculate Jaccard similarity of keywords
        if len(kw1) == 0 or len(kw2) == 0:
            return True

        jaccard = len(kw1 & kw2) / len(kw1 | kw2)

        # If keyword overlap > 70%, likely variants
        return jaccard > 0.70

    # Apply to same-category pairs
    same_category['is_variant'] = same_category.apply(are_title_variants, axis=1)
    interesting_same_cat = same_category[~same_category['is_variant']].copy()

    print(f"Interesting same-category pairs (after filtering variants): {len(interesting_same_cat)}")

    # Combine results
    if prioritize_cross_category:
        # Show cross-category first, then interesting same-category
        result = pd.concat([
            cross_category.sort_values('similarity', ascending=False),
            interesting_same_cat.sort_values('similarity', ascending=False)
        ])
    else:
        result = pd.concat([cross_category, interesting_same_cat]).sort_values('similarity', ascending=False)

    # Apply minimum similarity threshold
    result = result[result['similarity'] >= min_similarity]

    print(f"\nFinal interesting pairs: {len(result)}")

    # Save
    result.to_csv(output_csv, index=False)

    return result

def print_interesting_pairs(df: pd.DataFrame, top_n: int = 50):
    """Print interesting correlation pairs."""
    print(f"\n{'='*100}")
    print(f"TOP {min(top_n, len(df))} INTERESTING CROSS-MARKET CORRELATIONS")
    print(f"{'='*100}")

    for idx, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        cross_cat = "🔄 CROSS-CATEGORY" if row['cross_category'] else "📊 SAME CATEGORY"

        print(f"\n{idx}. Similarity: {row['similarity']:.4f} {cross_cat}")
        print(f"   [{row['ticker1']}] {row['category1']}")
        print(f"   \"{row['title1']}\"")
        print(f"   Volume: ${row['volume1']:,.0f}")
        print(f"   ↔")
        print(f"   [{row['ticker2']}] {row['category2']}")
        print(f"   \"{row['title2']}\"")
        print(f"   Volume: ${row['volume2']:,.0f}")
        print(f"   " + "-"*96)

def analyze_cross_category_patterns(df: pd.DataFrame):
    """Analyze cross-category correlation patterns."""
    cross_cat = df[df['cross_category']]

    if len(cross_cat) == 0:
        print("\nNo cross-category pairs found.")
        return

    print(f"\n{'='*100}")
    print(f"CROSS-CATEGORY CORRELATION PATTERNS ({len(cross_cat)} pairs)")
    print(f"{'='*100}")

    # Create category pair labels
    cross_cat = cross_cat.copy()
    cross_cat['category_pair'] = cross_cat.apply(
        lambda row: f"{row['category1']} ↔ {row['category2']}",
        axis=1
    )

    patterns = cross_cat['category_pair'].value_counts()

    for pattern, count in patterns.items():
        avg_sim = cross_cat[cross_cat['category_pair'] == pattern]['similarity'].mean()
        print(f"  {pattern}: {count} pairs (avg similarity: {avg_sim:.4f})")

def main():
    # Filter for interesting correlations
    interesting_df = filter_interesting_correlations(
        'encoding/cross_market_similarities.csv',
        'encoding/interesting_correlations.csv',
        min_similarity=0.70,
        prioritize_cross_category=True
    )

    # Print results
    print_interesting_pairs(interesting_df, top_n=50)

    # Analyze patterns
    analyze_cross_category_patterns(interesting_df)

    print(f"\n{'='*100}")
    print(f"Saved {len(interesting_df)} interesting pairs to encoding/interesting_correlations.csv")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()