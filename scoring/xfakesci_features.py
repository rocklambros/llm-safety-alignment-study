#!/usr/bin/env python3
"""
xFakeSci bigram network features per Hamed & Wu (2024).

Extracts textual fingerprint features based on bigram network analysis.
These features help distinguish AI-generated text from human text.

Reference: Hamed, A. A., & Wu, X. (2024). xFakeSci: A framework for
detecting fake scientific text using bigram network features.

Features:
    - nodes: Unique word count in the text
    - edges: Unique bigram (word pair) count
    - ratio: Edge-to-node ratio (network density proxy)
    - lcc_size: Largest connected component size
    - bigram_contrib: Bigram contribution ratio (repetition indicator)

Usage:
    features = extract_xfakesci_features("some text here")
    batch_features = extract_xfakesci_batch(["text1", "text2", ...])
"""

import re
from typing import Dict, List
import networkx as nx
from tqdm import tqdm


def extract_xfakesci_features(text: str) -> Dict[str, float]:
    """
    Extract bigram network features from text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with:
            - nodes: Number of unique words
            - edges: Number of unique bigrams
            - ratio: edges / nodes (network density)
            - lcc_size: Size of largest connected component
            - bigram_contrib: Total edge weight / bigram positions
    """
    # Handle empty/invalid input
    if not text or len(text.strip()) == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "ratio": 0.0,
            "lcc_size": 0,
            "bigram_contrib": 0.0
        }

    # Tokenize: lowercase, extract alphabetic words only
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # Handle single word or empty tokenization
    if len(words) < 2:
        return {
            "nodes": len(set(words)),
            "edges": 0,
            "ratio": 0.0,
            "lcc_size": len(set(words)),
            "bigram_contrib": 0.0
        }

    # Build weighted bigram network
    G = nx.Graph()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if G.has_edge(w1, w2):
            G[w1][w2]["weight"] += 1
        else:
            G.add_edge(w1, w2, weight=1)

    # Extract features
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "ratio": 0.0,
            "lcc_size": 0,
            "bigram_contrib": 0.0
        }

    # Edge-to-node ratio (network density proxy)
    ratio = n_edges / n_nodes

    # Largest connected component size
    if n_nodes > 0:
        connected_components = list(nx.connected_components(G))
        lcc_size = len(max(connected_components, key=len))
    else:
        lcc_size = 0

    # Bigram contribution ratio
    # Total edge weight divided by number of bigram positions
    # Higher values indicate more repetitive bigram patterns
    total_weight = sum(d["weight"] for _, _, d in G.edges(data=True))
    bigram_positions = len(words) - 1
    bigram_contrib = total_weight / bigram_positions if bigram_positions > 0 else 0.0

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "ratio": round(ratio, 4),
        "lcc_size": lcc_size,
        "bigram_contrib": round(bigram_contrib, 4)
    }


def extract_xfakesci_batch(
    texts: List[str],
    show_progress: bool = True
) -> List[Dict[str, float]]:
    """
    Extract xFakeSci features for a batch of texts.

    Args:
        texts: List of texts to analyze
        show_progress: Whether to show progress bar

    Returns:
        List of feature dictionaries
    """
    results = []

    iterator = texts
    if show_progress:
        iterator = tqdm(texts, desc="xFakeSci features", unit="text")

    for text in iterator:
        results.append(extract_xfakesci_features(text))

    return results


def get_network_stats(text: str) -> Dict:
    """
    Get detailed network statistics for analysis/debugging.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with detailed network statistics
    """
    if not text or len(text.strip()) == 0:
        return {"error": "empty_text"}

    words = re.findall(r'\b[a-z]+\b', text.lower())

    if len(words) < 2:
        return {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "bigram_count": 0,
            "network_built": False
        }

    # Build network
    G = nx.Graph()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if G.has_edge(w1, w2):
            G[w1][w2]["weight"] += 1
        else:
            G.add_edge(w1, w2, weight=1)

    # Compute various statistics
    stats = {
        "word_count": len(words),
        "unique_words": len(set(words)),
        "bigram_positions": len(words) - 1,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "num_components": nx.number_connected_components(G),
        "is_connected": nx.is_connected(G),
        "network_built": True
    }

    if G.number_of_nodes() > 0:
        # Top bigrams by weight
        edge_weights = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        stats["top_bigrams"] = edge_weights[:5]

        # Degree distribution summary
        degrees = [d for _, d in G.degree()]
        stats["max_degree"] = max(degrees)
        stats["min_degree"] = min(degrees)

    return stats


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I hate you and I hate everyone and I hate everything.",
        "This is a test. This is only a test. This is a test.",
        "Hello world!",
        "",  # Empty string
    ]

    print("=== xFakeSci Feature Extraction ===\n")

    for text in test_texts:
        features = extract_xfakesci_features(text)
        print(f"Text: {text[:50]!r}")
        print(f"  Nodes: {features['nodes']}")
        print(f"  Edges: {features['edges']}")
        print(f"  Ratio: {features['ratio']:.4f}")
        print(f"  LCC Size: {features['lcc_size']}")
        print(f"  Bigram Contrib: {features['bigram_contrib']:.4f}")
        print()

    print("=== Batch Processing ===")
    batch_results = extract_xfakesci_batch(test_texts[:3], show_progress=False)
    for text, features in zip(test_texts[:3], batch_results):
        print(f"{text[:30]!r}: ratio={features['ratio']:.4f}")

    print("\n=== Detailed Network Stats ===")
    detailed = get_network_stats(test_texts[2])  # Repetitive text
    print(f"Text: {test_texts[2]!r}")
    for k, v in detailed.items():
        print(f"  {k}: {v}")
