"""
Build a weighted, undirected student similarity graph from a similarity
matrix using either:
  - a similarity threshold (edge if sim >= threshold), or
  - a k-nearest-neighbours rule (edge to each student's top-k most similar
    peers; union of both directions).
"""

import networkx as nx
import numpy as np


def build_graph(similarity_matrix: np.ndarray, student_ids, mode="threshold",
                 threshold=0.25, k=5):
    G = nx.Graph()
    n = len(student_ids)
    for sid in student_ids:
        G.add_node(sid)

    if mode == "threshold":
        for i in range(n):
            for j in range(i + 1, n):
                w = similarity_matrix[i, j]
                if w >= threshold:
                    G.add_edge(student_ids[i], student_ids[j], weight=float(w))
    else:  # knn
        for i in range(n):
            row = similarity_matrix[i]
            top_k_idx = np.argsort(row)[::-1][:k]
            for j in top_k_idx:
                if i == j:
                    continue
                w = row[j]
                if w <= 0:
                    continue
                if G.has_edge(student_ids[i], student_ids[j]):
                    existing = G[student_ids[i]][student_ids[j]]["weight"]
                    G[student_ids[i]][student_ids[j]]["weight"] = max(existing, float(w))
                else:
                    G.add_edge(student_ids[i], student_ids[j], weight=float(w))

    return G
