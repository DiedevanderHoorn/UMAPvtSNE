import json 
import numpy as np
import os
from metrics import metric_stress, metric_shepard_diagram_correlation
from zadu.measures import *
from scipy import spatial
from collections import defaultdict
from utils import load_data
from sklearn.preprocessing import MinMaxScaler
from main import save_results, load_results


def load_embedding(path):
    with open(path, 'r') as f:
        data = json.load(f)
    embedding = np.array([[point["x"], point["y"]] for point in data['embedding']])
    labels = np.array(int(l) for l in data['labels'])
    return embedding, labels

def calculate_metrics(X, X_embedded, perplexity):
    trustworthiness, continuity = trustworthiness_continuity.measure(X, X_embedded, k=perplexity).values()
    neighb_hit = list(neighborhood_hit.measure(X_embedded, y, k=perplexity).values())[0]
   
    D_high_list = spatial.distance.pdist(X, "euclidean")
    high = max(D_high_list)
    D_high_list = D_high_list/high
    D_low_list = spatial.distance.pdist(X_embedded, "euclidean")
    low = max(D_low_list)
    D_low_list = D_low_list/low
    shepard_goodness = metric_shepard_diagram_correlation(D_high_list, D_low_list)
    norm_stress = metric_stress(D_high_list, D_low_list)

    return [trustworthiness, continuity, neighb_hit, shepard_goodness, norm_stress,]       

if __name__ == "__main__":
    results = defaultdict(dict, load_results('metricswithFS_newstress'))

    for dataset in os.listdir("data"):
        if dataset == 'digits':
            continue
        print(dataset)
        X, y = load_data('data', dataset)
        X = MinMaxScaler().fit_transform(X)
        UMAPUMAP_emb, labels = load_embedding(f"embeddings_best/{dataset}_UMAP-UMAP")
        results[dataset]["('UMAP', 'UMAP')"] = calculate_metrics(X, UMAPUMAP_emb, 15)
        tSNEtSNE_emb, labels = load_embedding(f"embeddings_best/{dataset}_t-SNE")
        results[dataset]["('t-SNE', 't-SNE')"] = calculate_metrics(X, tSNEtSNE_emb, 15)
        UMAPtSNE_emb, labels = load_embedding(f"embeddings_best/{dataset}_UMAP")
        results[dataset]["('UMAP', 't-SNE')"] = calculate_metrics(X, UMAPtSNE_emb, 15)
        LSPtSNE_emb, labels = load_embedding(f"embeddings_best/{dataset}_LSP")
        results[dataset]["('LSP', 't-SNE')"] = calculate_metrics(X, LSPtSNE_emb, 15)

        FS_emb, labels = load_embedding(f"embeddings_best/{dataset}_FS")
        results[dataset]['FS'] = calculate_metrics(X, FS_emb, 15)

        save_results(results, 'metricswithFS_newstress') 

    
