"""
Louvain community detection.

Primary path: python-louvain (the `community` package, imported as
`community_louvain`), which is the standard, widely-taught implementation.

Fallback: NetworkX's built-in `nx.community.louvain_communities` (available
in networkx>=3.0) is used automatically if python-louvain is not installed,
so the app never crashes for lack of one specific package. Both implement
the same Louvain algorithm (greedy modularity optimization).
"""

import networkx as nx

try:
    import community as community_louvain
    _HAS_PYTHON_LOUVAIN = True
except Exception:
    _HAS_PYTHON_LOUVAIN = False


def run_louvain(G: nx.Graph, resolution=1.0, random_state=42):
    """Returns (partition: dict node->community_id, modularity: float)."""
    if G.number_of_edges() == 0:
        # No edges at all -> every node is its own singleton community
        partition = {node: i for i, node in enumerate(G.nodes())}
        return partition, 0.0

    if _HAS_PYTHON_LOUVAIN:
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=random_state
        )
        modularity = community_louvain.modularity(partition, G, weight="weight")
    else:
        communities = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=random_state
        )
        partition = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = cid
        modularity = nx.community.modularity(G, communities, weight="weight")

    return partition, modularity


def community_sizes(partition):
    sizes = {}
    for node, cid in partition.items():
        sizes[cid] = sizes.get(cid, 0) + 1
    return sizes
