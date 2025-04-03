# %% [markdown]
# # Inferring transition probabilities from time-resolved single-cell data
#
# Here, we show how to infer cell-cell transition probabilities, using the RealTimeKernel. The required data can be downloaded [here](https://figshare.com/ndownloader/files/53395853) from FigShare. The data is expected to be in `data/larry/processed/`, _i.e._,
#
# ```bash
# mkdir -p ../../data/larry/processed/
# wget https://figshare.com/ndownloader/files/53395853 -O ../../data/larry/processed/adata.h5ad
# ```

# %% [markdown]
# ## Library imports

# %%
import matplotlib.pyplot as plt
import mplscience

import anndata as ad
import cellrank as cr
import scanpy as sc
import scvelo as scv
from moscot.problems.time import TemporalProblem

from crp import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
cr.settings.verbosity = 4
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "larry"

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "svg"

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_zarr(DATA_DIR / DATASET / "processed" / "adata.zarr")
adata

# %%
sc.pp.subsample(adata, fraction=0.25, random_state=0)
adata

# %% [markdown]
# ## Data preprocessing

# %%
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1500, subset=True)
adata

# %%
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="latent_rep", color="day", color_map="viridis", size=25, title="", ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="latent_rep", color="cell_type", size=25, title="", ax=ax)
    plt.show()

# %% [markdown]
# ## CellRank pipeline

# %%
adata.obs["day"] = adata.obs["day"].astype("category")

# %%
tp = TemporalProblem(adata)
tp = tp.prepare(time_key="day")

# %%
tp = tp.solve(epsilon=1e-3, tau_a=0.95, scale_cost="mean")

# %%
rtk = cr.kernels.RealTimeKernel.from_moscot(tp)
rtk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")
