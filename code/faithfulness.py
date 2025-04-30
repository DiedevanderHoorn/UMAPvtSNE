import networkx as nx
import numpy as np
def containment(G1, G2):
    return len(nx.intersection(G1, G2).edges) / len(G2.edges)

def faithfulness_graph_topologies(G1, G2):
    return len(nx.intersection(G1, G2).edges) / len(nx.compose(G1, G2).edges)

def normalize_by_max(G):
    max_weight_G = max(data["weight"] for _, _, data in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        data["weight"] /= max_weight_G
    return G

def weighted_jaccard(G1_unnorm, G2_unnorm):
    G1 = normalize_by_max(G1_unnorm)
    G2 = normalize_by_max(G2_unnorm)

    for G in [G1, G2]:
        weights = nx.get_edge_attributes(G, 'weight')
        # Get the min and max of the weights
        min_weight = min(weights.values())
        max_weight = max(weights.values())
        print(min_weight, max_weight)

    edges = set(G1.edges()).union(G2.edges())
    mins, maxs = 0, 0
    for edge in edges:
        weight1 = G1.get_edge_data(*edge, {}).get('weight', 0)
        weight2 = G2.get_edge_data(*edge, {}).get('weight', 0)
        mins += min(weight1, weight2)
        maxs += max(weight1, weight2)
    return mins / maxs



# def weighted_jaccard(G1_unnorm, G2_unnorm):
#     G1 = normalize_by_max(G1_unnorm)
#     G2 = normalize_by_max(G2_unnorm)
#     edges_G1 = set(G1.edges())
#     edges_G2 = set(G2.edges())

#     # Compute weighted intersection: sum of min weights
#     weighted_intersection = sum(
#         min(G1[u][v]["weight"], G2[u][v]["weight"])
#         for u, v in edges_G1 & edges_G2  # Only common edges
#     )
#     print(weighted_intersection)

#     all_edges = edges_G1 | edges_G2  # Union of edges
#     weighted_union = sum(
#         max(G1[u][v]["weight"], G2[u][v]["weight"]) if (u, v) in edges_G1 and (u, v) in edges_G2 
#         else G1[u][v]["weight"] if (u, v) in edges_G1 
#         else G2[u][v]["weight"] 
#         for u, v in all_edges
#     )
#     print(weighted_union)

#     weighted_jaccard_similarity = weighted_intersection / weighted_union if weighted_union != 0 else 0
#     return weighted_jaccard_similarity
   
