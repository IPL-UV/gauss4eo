import pandas as pd
import seaborn as sns
import numpy as np

sns.set_context(context="talk", font_scale=0.7)


def get_pivot_df(df: pd.DataFrame, stats_model: str = "pearson") -> pd.DataFrame:
    df = df[["model_0", "model_1", stats_model.lower()]]
    df = pd.concat(
        [df, df.rename(columns={"model_0": "model_1", "model_1": "model_0"})], axis=0
    ).drop_duplicates()
    df = df.pivot("model_0", "model_1", stats_model.lower())

    return df


def plot_heatmap(df, ax, **kwargs):
    sns.heatmap(df, ax=ax, **kwargs)
    ax.set(xlabel="", ylabel="")
    return ax


def plot_dendrogram(df: pd.DataFrame, ax, names):

    names = [i for i in names.values()]

    import scipy.cluster.hierarchy as spc

    pdist = 1 - np.abs(df.values)
    pdist = pdist[np.triu_indices_from(pdist, k=1)]
    linkage = spc.linkage(pdist, method="complete", optimal_ordering=True)
    order = spc.leaves_list(linkage)
    spc.dendrogram(
        linkage,
        above_threshold_color="k",
        color_threshold=0.4,
        orientation="right",
        leaf_label_func=lambda i: names[i],
        leaf_font_size=10,
        leaf_rotation=0,
        ax=ax,
    )
    return ax
