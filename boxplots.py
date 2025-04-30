from main import load_results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches

def create_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    box_plot = ax.boxplot(df.values.T, patch_artist=True, tick_labels=df.index)

    colormap = plt.get_cmap('tab10')
    xlabels = LabelEncoder().fit_transform(df.index.unique())  
    colors = [colormap.colors[i] for i in xlabels]


    for patch, color in zip(box_plot['boxes'], colors):  
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    for median in box_plot['medians']:  
        median.set_color('black')
        median.set_linewidth(2)

    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.set_ylabel("Faithfulness", fontsize = 15)
    ax.set_xticklabels(["t-SNE - UMAP", "LSP - t-SNE", "LSP - UMAP"], fontsize=15)
    ax.tick_params(axis='y', labelsize = 12)

    fig.savefig(f'boxplots_best.png', dpi=400)
    plt.show()

def create_barchart(df):
    ax = df.plot(kind="bar", figsize=(10, 6), colormap="tab20")
    ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.ylabel("Faithfulness", fontsize=12)
    plt.xticks(rotation=0)

    plt.savefig("barcharts.png")

def create_boxplot_metrics(df):
    import pandas as pd
    import matplotlib.pyplot as plt

    metric_names = ['trustworthiness', 'continuity', 'neighb_hit', 'shepard goodness', 'stress']

    # Collect exploded data
    records = []

    df.reset_index()
    for index, row in df.iterrows():
        technique = index  # first column is the technique tuple

        for dataset_name in df.columns:  # skip the first column
            metric_values = row[dataset_name]  # this should be a list of metric values
            for metric, value in zip(metric_names, metric_values):
                if metric == 'stress':
                    value = 1 - value
                records.append({
                    'technique': technique,
                    'dataset': dataset_name,
                    'metric': metric,
                    'value': value
                })

    full_df = pd.DataFrame(records)
    print(full_df)
    long_df = full_df[full_df.technique != 'FS']
    print(full_df.groupby(['technique', 'metric']).mean(numeric_only=True))

    # Get unique techniques and metrics
    techniques = long_df['technique'].unique()
    metrics = long_df['metric'].unique()

    # Assign a color per metric
    metric_colors = {
        'trustworthiness': '#ff7f0e',
        'continuity': '#2ca02c',
        'neighb_hit': '#d62728',
        'shepard goodness': '#9467bd',
        'stress': '#1f77b4'
    }

    # Create plot data: one box per (technique, metric)
    plot_data = []
    positions = []
    colors = []
    labels = []

    pos = 0
    for tech in techniques:
        for metric in metrics:
            values = long_df[(long_df['technique'] == tech) & (long_df['metric'] == metric)]['value'].tolist()
            plot_data.append(values)
            positions.append(pos)
            colors.append(metric_colors[metric])
            labels.append(f'{tech}')
            pos += 1

    # Create figure
    plt.figure(figsize=(max(12, len(plot_data) * 0.5), 6))

    for i, (vals, color) in enumerate(zip(plot_data, colors)):
        bplot = plt.boxplot(
            vals,
            positions=[positions[i]],
            widths=0.6,
            patch_artist=True,
            showfliers=False
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

        for median in bplot['medians']:  
            median.set_color('black')
            median.set_linewidth(2)

    metrics_per_group = len(metric_colors)

    group_centers = [
        sum(positions[i:i+metrics_per_group]) / metrics_per_group
        for i in range(0, len(positions), metrics_per_group)
    ]

    technique_labels = []
    for label in labels:
        if label not in technique_labels:
            technique_labels.append(label)
    technique_labels = ['r: UMAP, m: UMAP', 'r: t-SNE, m: t-SNE', 'r: UMAP, m: t-SNE', 'r: LSP, m: t-SNE']
    plt.xticks(group_centers, technique_labels, rotation=0, ha='center', fontsize = 15)

    metric_dict = {'trustworthiness' : 'trustworthiness' , 
                   'continuity' : 'continuity', 
                   'neighb_hit' : 'neighborhood hit', 
                   'shepard goodness' : 'shepard goodness', 
                   'stress' : '1 - stress'}
    # Create a custom legend
    legend_patches = [mpatches.Patch(color=color, label=metric_dict[metric]) for metric, color in metric_colors.items()]
    plt.legend(handles=legend_patches, title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15, title_fontsize=15)

    ax = plt.gca()
    ax.spines[['right', 'top', 'left']].set_visible(False)

    plt.yticks(fontsize=15)
    plt.ylabel("Score", fontsize=17)
    plt.tight_layout()
    plt.savefig("metric_boxplots_final_fontincreased_newstress.png", dpi=400)

if __name__ == "__main__":
    result_dict = load_results("metricswithFS_newstress")
    datasets = result_dict.keys()
    df = pd.DataFrame(result_dict, index=None)

    create_boxplot_metrics(df)
    #create_barchart(df)
    #create_boxplot(df)
    





   
