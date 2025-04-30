# Reviewer #2: "Why didn't you use UMAP?"
Supplementary material for the extended abstract and corresponding poster, titled: Reviewer #2: ''Why didn't you use UMAP?'', authors: D.P.M. van der Hoorn, A. Arleo, F.V. Paulovich, to be presented at EuroVis 2025.

Source code can be found on [GitHub](https://github.com/DiedevanderHoorn/UMAPvtSNE).

## Additional results
To find the projections for all combinations, consider the _figures best_ folders. 

### Folders
- The data folder contains all datasets used in numpy format. These datasets were obtained through: https://mespadoto.github.io/proj-quant-eval/post/datasets/.
- The embeddings folder contains all embeddings used for the results.
- The figures folders contain projections on all datasets in different configurations (combination of relationship/mapping phase).

## How to recreate the results
The file called "perplexities" contains all the perplexities used for t-SNE to maximize faitfulness. 
- To get the projections: main.py
- To get metrics: get_metrics.py
- To get the boxplots: boxplots.py