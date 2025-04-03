# %% [markdown]
# # Human hematopoiesis analysis with the CytoTRACEKernel
#
# In this analysis, we use the CytoTRACEKernel to recover human bone marrow development. To run the analysis, ensure that
# the processed data is either already saved, or generate it using the corresponding notebook in the `pseudotime/`
# directory.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import anndata as ad
import cellrank as cr
import scanpy as sc
import scvelo as scv

from crp import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
cr.settings.verbosity = 4
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "cytotracekernel").mkdir(parents=True, exist_ok=True)

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

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / "bone_marrow" / "processed" / "adata.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 6))
    scv.pl.scatter(adata, basis="umap", color=["celltype"], size=25, dpi=100, title="", ax=ax)
    plt.show()

# %%
adata.layers["spliced"] = adata.X
adata.layers["unspliced"] = adata.X

scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %% [markdown]
# ## CytoTRACEKernel

# %% [markdown]
# ### Kernel

# %%
ctk = cr.kernels.CytoTRACEKernel(adata)
ctk.compute_cytotrace()

# %%
with mplscience.style_context():
    scv.pl.scatter(
        adata, basis="umap", color=["celltype", "ct_pseudotime"], legend_loc="right", color_map="viridis", size=25
    )
    plt.show()

# %%
ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.5)

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ctk)
estimator.compute_schur(n_components=20)
with mplscience.style_context():
    if SAVE_FIGURES:
        save = FIG_DIR / "cytotracekernel" / f"spectrum.{FIGURE_FORMAT}"
    else:
        save = False
    estimator.plot_spectrum(real_only=True, figsize=(6, 3), save=save)
    plt.show()

# %%
estimator.compute_macrostates(4, cluster_key="celltype")

with mplscience.style_context():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="all", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "cytotracekernel" / "4_macrostates.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %%
estimator.compute_macrostates(7, cluster_key="celltype")

with mplscience.style_context():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="all", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "cytotracekernel" / "7_macrostates.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %%
estimator.compute_macrostates(15, cluster_key="celltype")

with mplscience.style_context():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="all", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "cytotracekernel" / "15_macrostates.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %%
cluster_key = "celltype"

tsi_score = estimator.tsi(n_macrostates=15, terminal_states=TERMINAL_STATES, cluster_key=cluster_key)
print(f"TSI score: {tsi_score:.2f}")

if SAVE_RESULTS:
    estimator._tsi.to_df().to_parquet(DATA_DIR / "bone_marrow" / "results" / "ct_tsi.parquet")

# %%
palette = {"CytoTRACEKernel": "#DE8F05", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    save = FIG_DIR / "cytotracekernel" / f"tsi_ranking.{FIGURE_FORMAT}"
else:
    save = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette, save=save)
    plt.show()

# %%
estimator.set_terminal_states(["Ery_1", "CLP", "pDC_1", "cDC_1", "Mono"])
estimator.rename_terminal_states({"Ery_1": "Ery", "pDC_1": "pDC", "cDC_1": "cDC"})

with mplscience.style_context():
    estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", title="", size=100)

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 6))
        estimator.plot_macrostates(which="terminal", basis="umap", legend_loc=False, title="", size=100, ax=ax)
        fig.savefig(
            FIG_DIR / "cytotracekernel" / "terminal_states.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

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
                FIG_DIR / "cytotracekernel" / f"{lineage.lower()}_fate_terminal_set.png",
                dpi=400,
                transparent=True,
                bbox_inches="tight",
            )
            plt.show()
    del bdata

# %%
cluster_key = "celltype"
rep = "X_pca"

cbc = []
for source, target in tqdm(STATE_TRANSITIONS):
    _cbc = ctk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    cbc.append(
        pd.DataFrame(
            {
                "state_transition": [f"{source} - {target}"] * len(_cbc),
                "cbc": _cbc,
            }
        )
    )
cbc = pd.concat(cbc)

if SAVE_RESULTS:
    cbc.to_parquet(DATA_DIR / "bone_marrow" / "results" / "ct_cbc.parquet")

cbc.head()
