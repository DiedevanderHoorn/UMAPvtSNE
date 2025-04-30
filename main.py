import os
import json
from itertools import combinations
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils import load_data
from umap_gd import umap_graph
from tsne_gd import tsne_bh_prob_graph
from lsp_gd import lsp_graph
from faithfulness import faithfulness_graph_topologies, weighted_jaccard
from t_sne_bridging_dr_gd import draw_graph_by_tsne
import matplotlib.pyplot as plt
from utils import store_projection
import umap
from force_scheme import ForceScheme


def load_results(filename = "faithfulness_best_results"):
    try:
        with open(filename, "r") as f:
            return json.load(f)  # Load as dictionary
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Return empty dict if file does not exist or is corrupted

def save_results(data, filename="faithfulness_best_results"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4) 

def run_and_store_umap(X, label, dataset):
    y = umap.UMAP(n_neighbors=15,
                   metric="euclidean",
                   random_state=42).fit_transform(X)
    store_projection(y, [str(l) for l in label], f'embeddings_best/{dataset}_{"UMAP-UMAP"}')

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
    plt.title(f'Relationship phase: UMAP, Mapping phase: UMAP')
    plt.savefig(f'figures_best_higher_res/{dataset}_UMAP-UMAP', dpi=300)
    plt.close()

if __name__ == "__main__":
    results = defaultdict(dict, load_results())
    dataset_perp_dict = defaultdict(dict)

    datasets = os.listdir('data')
    for dataset in datasets:
        if dataset in results:
           print(f'skipping {dataset}')
           continue
        
        print(dataset)
        X, y = load_data('data', dataset)
        X = MinMaxScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        X_emb = ForceScheme(max_it=500).fit_transform(X)
        store_projection(X_emb, [str(l) for l in y], f'embeddings_best/{dataset}_FS')

        #run full UMAP algorithm
        run_and_store_umap(X, y, dataset)

        def generate_function_dict(n_neighbors = 15):
            return {
                'UMAP' : umap_graph(X, nr_neighbors=15, metric='euclidean', labels=y),
                't-SNE' : tsne_bh_prob_graph(X, perplexity = n_neighbors, metric='euclidean', labels = y, epsilon=0.9),
                'LSP' : lsp_graph(X, nr_neighbors = 15, metric='euclidean', labels = y)
            }

        function_dict = generate_function_dict()
        options = ['UMAP', 't-SNE', 'LSP']
        combination_list = [('UMAP', 't-SNE'),
                            ('LSP', 't-SNE'),
                            ('UMAP', 'LSP')]
        for tech1, tech2 in combination_list:
            if tech2 == 't-SNE':
                if dataset not in dataset_perp_dict.keys():
                    G1 = function_dict[tech1]
                    best_score = 0
                    best_perp = 0
                    for perp in list(range(2, 24, 2)):
                        G2 = tsne_bh_prob_graph(X, perplexity = perp, metric='euclidean', labels = y, epsilon=0.9)
                        score = faithfulness_graph_topologies(G1, G2)
                        if score > best_score:
                            best_score = score
                            best_perp = perp
                results[dataset][str((tech1, tech2))] = best_score
                dataset_perp_dict[dataset] = best_perp
            else:
                G1 = function_dict[tech1]
                G2 = function_dict[tech2]
                results[dataset][str((tech1, tech2))] = faithfulness_graph_topologies(G1, G2)
        save_results(results)
        save_results(dataset_perp_dict, "perplexities")

        for technique in options:
            if technique != 't-SNE':
                G = function_dict[technique]
            else:
                G = tsne_bh_prob_graph(X, perplexity = dataset_perp_dict[dataset], metric='euclidean', labels = y, epsilon=0.9)
            X_emb, label = draw_graph_by_tsne(X, G)
            store_projection(X_emb, label, f'embeddings_best/{dataset}_{technique}')

            plt.figure()
            plt.scatter(X_emb[:, 0], X_emb[:, 1], c=label, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
            plt.xticks([])
            plt.yticks([])
            ax = plt.gca()
            ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
            plt.title(f'Relationship phase: {technique}, Mapping phase: t-SNE')
            plt.savefig(f'figures_best_higher_res/{dataset}_{technique}', dpi=300)
            plt.close()
        run_and_store_umap(X, label, dataset)

