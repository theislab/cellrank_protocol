# %% [markdown]
# # Human hematopoiesis analysis with the PseudotimeKernel
#
# In this analysis, we recapitulate human bone marrow development with the PseudotimeKernel, relying on pseudotime
# estimates computed with Palantir. To run the analysis, ensure that the processed data is either already saved, or
# generate it using the corresponding notebook `preprocessing.ipynb`.
#
# The corresponding data can be generated through the preprocessing notebooks `pseudotime/preprocessing.ipynb`, or
# downloaded driectly from [here](https://figshare.com/ndownloader/files/53394941) and should be
# saved in `data/bone_marrow/processed/adata.h5ad`, _i.e._,
#
# ```bash
# mkdir -p ../../data/bone_marrow/processed/
# wget https://figshare.com/ndownloader/files/53394941 -O ../../data/bone_marrow/processed/adata.h5ad
# ```

# %% [markdown]
# ## Library imports

# %%
import warnings

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib.patches import Patch

import anndata as ad
import cellrank as cr
import scanpy as sc
import scvelo as scv

from crp import DATA_DIR, FIG_DIR
from crp.core import G2M_GENES, S_GENES
from crp.plotting import plot_tsi

# %% [markdown]
# ## General settings

# %%
warnings.filterwarnings(action="ignore", category=FutureWarning)

# %%
sc.settings.verbosity = 2
cr.settings.verbosity = 4
scv.settings.verbosity = 3

# %%
mpl.use("module://matplotlib_inline.backend_inline")
mpl.rcParams["backend"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "pseudotimekernel").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "svg"

# %%
SAVE_RESULTS = True
if SAVE_RESULTS:
    (DATA_DIR / "bone_marrow" / "results").mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Ery", "CLP", "cDC", "pDC", "Mono"]

# %%
STATE_TRANSITIONS = [
    ("HSC", "MEP"),
    ("MEP", "Ery"),
    ("HSC", "HMP"),
    ("HMP", "Mono"),
    ("HMP", "CLP"),
    ("HMP", "DCPre"),
    ("DCPre", "pDC"),
    ("DCPre", "cDC"),
]

# %%
SIGNIFICANCE_PALETTE = {"n.s.": "#dedede", "*": "#90BAAD", "**": "#A1E5AB", "***": "#ADF6B1"}


# %% [markdown]
# ## Function definitions


# %%
def get_significance(pvalue) -> str:
    """Assign significance symbol based on p-value."""
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# %%
def get_ttest_res(cbc: pd.DataFrame):
    """Perform t tests to assess CBC difference."""
    ttest_res = {}
    significances = {}

    for source, target in STATE_TRANSITIONS:
        obs_mask = cbc["State transition"].isin([f"{source} - {target}"])
        a = cbc.loc[obs_mask, "Log ratio"].values
        b = np.zeros(len(a))

        ttest_res[f"{source} - {target}"] = ttest_ind(a, b, equal_var=False, alternative="greater")
        significances[f"{source} - {target}"] = get_significance(ttest_res[f"{source} - {target}"].pvalue)

    return ttest_res, significances


# %%
def get_palette(significances: dict[str, float]) -> dict[str, str]:
    """Generate a color palette for statistical significance."""
    palette = {
        state_transition: SIGNIFICANCE_PALETTE[significance] for state_transition, significance in significances.items()
    }
    return palette


# %%
def plot_cbc_ratio(cbc: pd.DataFrame, palette: dict, figsize=(12, 6), fpath: None | str = None) -> None:
    """Plot the CBC ratio, colored by significance, to compare kernel performance."""
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(data=cbc, x="State transition", y="Log ratio", palette=palette, ax=ax)

        ax.tick_params(axis="x", rotation=45)

        handles = [Patch(label=label, facecolor=color) for label, color in SIGNIFICANCE_PALETTE.items()]
        fig.legend(
            handles=handles,
            labels=["n.s.", "p<1e-1", "p<1e-2", "p<1e-3"],
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.025),
        )
        fig.tight_layout()
        plt.show()

        if fpath is not None:
            fig.savefig(fpath, format="svg", transparent=True, bbox_inches="tight")


# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / "bone_marrow" / "processed" / "adata.h5ad")
adata

# %% [markdown]
# ## PseudotimeKernel

# %% [markdown]
# ### Kernel

# %%
ptk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
ptk.compute_transition_matrix(threshold_scheme="soft")

# %%
with mplscience.style_context():
    if SAVE_FIGURES:
        dpi = 400
        save = FIG_DIR / "pseudotimekernel" / "umap_rws.png"
    else:
        dpi = 150
        save = None
    ptk.plot_random_walks(
        start_ixs={"celltype": "HSC"}, basis="umap", seed=0, dpi=dpi, size=30, save=save, figsize=(6, 6)
    )

    plt.show()

# %%
with mplscience.style_context():
    ptk.plot_random_walks(start_ixs={"palantir_pseudotime": [0, 0.1]}, basis="umap", seed=0, dpi=150, size=30)
    plt.show()

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ptk)
estimator.compute_schur()
with mplscience.style_context():
    if SAVE_FIGURES:
        save = FIG_DIR / "pseudotimekernel" / f"spectrum.{FIGURE_FORMAT}"
    else:
        save = False
    estimator.plot_spectrum(real_only=True, figsize=(6, 3), save=save)
    plt.show()

# %%
estimator.compute_macrostates(5, cluster_key="celltype")

with mplscience.style_context():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="all", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "pseudotimekernel" / "5_macrostates.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

# %%
cluster_key = "celltype"

tsi_score = estimator.tsi(n_macrostates=15, terminal_states=TERMINAL_STATES, cluster_key=cluster_key)
print(f"TSI score: {tsi_score:.2f}")

# %%
palette = {"PseudotimeKernel": "#DE8F05", "Optimal identification": "#000000"}

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette)
    plt.show()

# %%
estimator.compute_macrostates(6, cluster_key="celltype")

with mplscience.style_context():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="all", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "pseudotimekernel" / "6_macrostates.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

# %%
estimator.predict_terminal_states()
# estimator.set_terminal_states(TERMINAL_STATES)

with mplscience.style_context():
    estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", title="", size=100)

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="terminal", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "pseudotimekernel" / "terminal_states_inferred.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %%
estimator.compute_fate_probabilities()
with mplscience.style_context():
    estimator.plot_fate_probabilities(same_plot=False, basis="umap", ncols=6)
    plt.show()

if SAVE_FIGURES:
    bdata = adata.copy()
    bdata.obs[estimator.fate_probabilities.names.tolist()] = estimator.fate_probabilities.X

    for lineage in estimator.fate_probabilities.names.tolist():
        with mplscience.style_context():
            fig, ax = plt.subplots(figsize=(6, 6))
            scv.pl.scatter(bdata, basis="umap", color=lineage, cmap="viridis", title="", colorbar=False, ax=ax)
            fig.savefig(
                FIG_DIR / "pseudotimekernel" / f"{lineage.lower()}_fate_terminal_pred.png",
                dpi=400,
                transparent=True,
                bbox_inches="tight",
            )
            plt.show()
    del bdata

# %%
if SAVE_FIGURES:
    save = FIG_DIR / "pseudotimekernel" / f"circular_projection.{FIGURE_FORMAT}"
else:
    save = None

cr.pl.circular_projection(adata, keys=cluster_key, legend_loc="right", save=save)

# %%
drivers = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="celltype", lineages=["MEP"], clusters=["HSC", "MEP"]
)

with mplscience.style_context():
    estimator.plot_lineage_drivers(lineage="MEP", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")
    plt.show()

# %%
gene_names = drivers.loc[
    ~(drivers.index.str.startswith(("MT.", "RPL", "RPS", "^HB[^(p)]")) | drivers.index.isin(S_GENES + G2M_GENES)),
    :,
].index

ranking = pd.DataFrame(drivers.loc[gene_names, "MEP_corr"])
ranking["ranking"] = np.arange(len(gene_names))

# %%
df = ranking.iloc[:20, :]

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    y_min = np.min(df["MEP_corr"])
    y_max = np.max(df["MEP_corr"])

    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.4 * (y_max - y_min)
    ax.set_ylim(y_min, y_max)

    ax.set_xlim(-0.75, 19.5)

    for gene in df.index:
        color = "#000000"
        ax.text(
            df.loc[gene, "ranking"],
            df.loc[gene, "MEP_corr"],
            gene,
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=20,
            color=color,
        )

    if SAVE_FIGURES:
        ax.set(xticks=[], xticklabels=[])
        fig.savefig(
            FIG_DIR / "pseudotimekernel" / f"putative_drivers.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

# %% [markdown]
# ITGA2B: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0043300, https://pmc.ncbi.nlm.nih.gov/articles/PMC7217381/, https://www.sciencedirect.com/science/article/pii/S1538783623003215
# VWF: https://pmc.ncbi.nlm.nih.gov/articles/PMC7217381/
# PLEK: https://www.sciencedirect.com/science/article/pii/S1097276520302343
#
# RGS18: https://pmc.ncbi.nlm.nih.gov/articles/PMC1222126/, https://pubmed.ncbi.nlm.nih.gov/25405900/
# PIEZO2: https://pmc.ncbi.nlm.nih.gov/articles/PMC5803306/
#
# SLC24A3: https://pmc.ncbi.nlm.nih.gov/articles/PMC2225987/ (platlet = meg derived)
# STXBP5: https://www.sciencedirect.com/science/article/pii/S1538783623003215
#
# Double check
# ARHGAP6: https://pmc.ncbi.nlm.nih.gov/articles/PMC6199649/

# %%
model = cr.models.GAM(adata)

with mplscience.style_context():
    cr.pl.gene_trends(
        adata,
        model=model,
        genes=["ITGA2B", "VWF", "PLEK", "RGS18", "PIEZO2"],
        time_key="palantir_pseudotime",
        hide_cells=True,
        same_plot=True,
    )
    plt.show()

if SAVE_FIGURES:
    fig = cr.pl.gene_trends(
        adata,
        model=model,
        genes=["ITGA2B", "VWF"],
        time_key="palantir_pseudotime",
        hide_cells=True,
        same_plot=True,
        figsize=(8, 4),
        sharey=True,
        return_figure=True,
        # lineage_cmap=["#8e063b", "#f0b98d", "#d5eae7", "#f3e1eb"],
    )

    fig.savefig(
        FIG_DIR / "pseudotimekernel" / f"gene_trends.{FIGURE_FORMAT}",
        format=FIGURE_FORMAT,
        transparent=True,
        bbox_inches="tight",
    )

# %%
estimator.set_terminal_states(TERMINAL_STATES)

with mplscience.style_context():
    estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", title="", size=100)

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="terminal", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "pseudotimekernel" / "terminal_states_set.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %%
ptk_tsi = estimator._tsi.to_df()
ct_tsi = pd.read_parquet(DATA_DIR / "bone_marrow" / "results" / "ct_tsi.parquet")

ptk_tsi["method"] = "PseudotimeKernel"
ct_tsi["method"] = "CytoTRACEKernel"

df = pd.concat([ptk_tsi, ct_tsi])

palette = {"PseudotimeKernel": "#0173B2", "CytoTRACEKernel": "#DE8F05", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fname = FIG_DIR / "pseudotimekernel" / f"tsi_ranking.{FIGURE_FORMAT}"
else:
    fname = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    plot_tsi(df=df, n_macrostates=15, palette=palette, fname=fname)
    plt.show()

# %%
palette = {"PseudotimeKernel": "#DE8F05", "CytoTRACEKernel": "#DE8F05", "Optimal identification": "#000000"}

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette)
    plt.show()

# %%
estimator.compute_fate_probabilities()
with mplscience.style_context():
    estimator.plot_fate_probabilities(same_plot=False, basis="umap", ncols=5)
    plt.show()

if SAVE_FIGURES:
    bdata = adata.copy()
    bdata.obs[estimator.fate_probabilities.names.tolist()] = estimator.fate_probabilities.X

    for lineage in estimator.fate_probabilities.names.tolist():
        with mplscience.style_context():
            fig, ax = plt.subplots(figsize=(6, 6))
            scv.pl.scatter(bdata, basis="umap", color=lineage, cmap="viridis", title="", colorbar=False, ax=ax)
            fig.savefig(
                FIG_DIR / "pseudotimekernel" / f"{lineage.lower()}_fate_terminal_set.png",
                dpi=400,
                transparent=True,
                bbox_inches="tight",
            )
            plt.show()
    del bdata

# %%
cluster_key = "celltype"
rep = "X_pca"

ptk_cbc = []
for source, target in tqdm(STATE_TRANSITIONS):
    _cbc = ptk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    ptk_cbc.append(
        pd.DataFrame(
            {
                "state_transition": [f"{source} - {target}"] * len(_cbc),
                "cbc": _cbc,
            }
        )
    )
ptk_cbc = pd.concat(ptk_cbc)

if SAVE_RESULTS:
    ptk_cbc.to_parquet(DATA_DIR / "bone_marrow" / "results" / "palantir_cbc.parquet")

ptk_cbc.head()

# %%
ctk_cbc = pd.read_parquet(DATA_DIR / "bone_marrow" / "results" / "ct_cbc.parquet")
cbc_ratio = pd.DataFrame(
    {"Log ratio": np.log((ptk_cbc["cbc"] + 1) / (ctk_cbc["cbc"] + 1)), "State transition": ptk_cbc["state_transition"]}
)

# %%
ttest_res, significances = get_ttest_res(cbc_ratio)
palette = get_palette(significances=significances)

if SAVE_FIGURES:
    fpath = FIG_DIR / "pseudotimekernel" / "cbc_ratio.svg"
else:
    fpath = None
plot_cbc_ratio(cbc=cbc_ratio, palette=palette, figsize=(12, 4), fpath=fpath)
