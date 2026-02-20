"""
Title Similarity Analysis using Sentence Transformers
Encodes market titles and finds most similar/dissimilar pairs.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def load_events(csv_path: str) -> pd.DataFrame:
    """Load events from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events from {csv_path}")
    return df

def create_full_titles(df: pd.DataFrame) -> list:
    """Combine title and subtitle for richer context."""
    full_titles = []
    for _, row in df.iterrows():
        title = row['title']
        subtitle = row.get('sub_title', '')

        if pd.notna(subtitle) and subtitle:
            full_title = f"{title} {subtitle}"
        else:
            full_title = title

        full_titles.append(full_title)

    return full_titles

def encode_titles(titles: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Encode titles using sentence transformer model."""
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(titles)} titles...")
    embeddings = model.encode(titles, show_progress_bar=True, convert_to_numpy=True)

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def find_extreme_pairs(embeddings: np.ndarray, titles: list, df: pd.DataFrame, top_k: int = 5):
    """Find most similar and most dissimilar pairs."""
    # Compute cosine similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    # Get upper triangle indices (avoid duplicates and self-similarity)
    n = len(titles)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'idx1': i,
                'idx2': j,
                'similarity': similarity_matrix[i, j],
                'title1': titles[i],
                'title2': titles[j],
                'ticker1': df.iloc[i]['event_ticker'],
                'ticker2': df.iloc[j]['event_ticker'],
                'category1': df.iloc[i]['category'],
                'category2': df.iloc[j]['category'],
                'volume1': df.iloc[i]['dollar_volume'],
                'volume2': df.iloc[j]['dollar_volume'],
            })

    pairs_df = pd.DataFrame(pairs)

    # Sort by similarity
    most_similar = pairs_df.nlargest(top_k, 'similarity')
    most_dissimilar = pairs_df.nsmallest(top_k, 'similarity')

    return most_similar, most_dissimilar, similarity_matrix

def print_pairs(pairs_df: pd.DataFrame, title: str):
    """Pretty print pairs."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    for idx, row in pairs_df.iterrows():
        print(f"\nSimilarity Score: {row['similarity']:.4f}")
        print(f"  [{row['ticker1']}] {row['category1']}")
        print(f"  \"{row['title1']}\"")
        print(f"  Volume: ${row['volume1']:,.0f}")
        print(f"  ---")
        print(f"  [{row['ticker2']}] {row['category2']}")
        print(f"  \"{row['title2']}\"")
        print(f"  Volume: ${row['volume2']:,.0f}")
        print(f"  {'-'*80}")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "market_data" / "kalshi_non_sports_events.csv"

    # Load events
    df = load_events(csv_path)

    # Create full titles
    full_titles = create_full_titles(df)

    # Encode titles
    embeddings = encode_titles(full_titles)

    # Find extreme pairs
    most_similar, most_dissimilar, similarity_matrix = find_extreme_pairs(
        embeddings, full_titles, df, top_k=5
    )

    # Print results
    print_pairs(most_similar, "TOP 5 MOST SIMILAR TITLE PAIRS")
    print_pairs(most_dissimilar, "TOP 5 MOST DISSIMILAR TITLE PAIRS")

    # Save embeddings and similarity matrix
    output_dir = base_dir / "encoding"
    np.save(output_dir / "title_embeddings.npy", embeddings)
    np.save(output_dir / "similarity_matrix.npy", similarity_matrix)

    # Save pairs to CSV
    most_similar.to_csv(output_dir / "most_similar_pairs.csv", index=False)
    most_dissimilar.to_csv(output_dir / "most_dissimilar_pairs.csv", index=False)

    print(f"\n{'='*80}")
    print("Files saved:")
    print(f"  - {output_dir / 'title_embeddings.npy'}")
    print(f"  - {output_dir / 'similarity_matrix.npy'}")
    print(f"  - {output_dir / 'most_similar_pairs.csv'}")
    print(f"  - {output_dir / 'most_dissimilar_pairs.csv'}")
    print(f"{'='*80}")

    # Stats
    print(f"\nSimilarity Statistics:")
    print(f"  Mean similarity: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.4f}")
    print(f"  Max similarity: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max():.4f}")
    print(f"  Min similarity: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min():.4f}")

if __name__ == "__main__":
    main()