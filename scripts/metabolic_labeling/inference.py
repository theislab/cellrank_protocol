# %% [markdown]
# # Metabolic-labeling based RNA velocity inference
#
# Metabolic-labeling based RNA velocity inference as proposed in
# [CellRank 2: unified fate mapping in multiview single-cell data](https://doi.org/10.1038/s41592-024-02303-9); the
# required data can be downloaded [here](https://figshare.com/ndownloader/files/56614223) from FigShare and is expected to
# be in `data/rpe1/processed/`, _i.e._,
#
# ```bash
# mkdir -p ../../data/rpe/processed/
# wget https://figshare.com/ndownloader/files/56614223 -O ../../data/rpe1/processed/adata.h5ad
# ```

# %% [markdown]
# ## Library imports

# %%
from scipy.sparse import csr_matrix

import anndata as ad
import scanpy as sc
import scvelo as scv
from scvelo.inference import (
    get_labeling_time_mask,
    get_labeling_times,
    get_n_neighbors,
    get_obs_dist_argsort,
    get_parameters,
)

from crp import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3

# %%
SAVE_DATA = True

# %%
SAVE_FIGURES = False

# %% [markdown]
# ## Constants

# %%
DATASET_ID = "rpe1"

if SAVE_DATA:
    (DATA_DIR / DATASET_ID / "results").mkdir(parents=True, exist_ok=True)

if SAVE_FIGURES:
    (FIG_DIR / DATASET_ID).mkdir(parents=True, exist_ok=True)

# %%
FIGURE_FORMAT = "pdf"

# %%
N_JOBS = 8

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET_ID / "processed" / "adata.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %%
adata = adata[adata.obs["labeling_time"] != "dmso", :].copy()
adata.obs["labeling_time"] = adata.obs["labeling_time"].astype(float) / 60

adata.obs.drop(["well_id", "plate_id", "log10_gfp", "log10_gfp"], axis=1, inplace=True)

del (
    adata.layers["labeled_unspliced"],
    adata.layers["labeled_spliced"],
    adata.layers["unlabeled_unspliced"],
    adata.layers["unlabeled_spliced"],
)

# %%
scv.pp.filter_and_normalize(adata, min_counts=50, layers_normalize=["X", "labeled", "unlabeled", "total"], log=False)
adata

# %%
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
adata

# %%
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

# %%
adata.layers["labeled_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["labeled"]).A
adata.layers["unlabeled_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["unlabeled"]).A
adata.layers["total_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["total"]).A

# %% [markdown]
# ## Parameter inference

# %%
time_key = "labeling_time"
labeling_times = get_labeling_times(adata=adata, time_key="labeling_time")

labeling_time_mask = get_labeling_time_mask(adata=adata, time_key=time_key, labeling_times=labeling_times)

obs_dist_argsort = get_obs_dist_argsort(adata=adata, labeling_time_mask=labeling_time_mask)

# %%
n_neighbors = get_n_neighbors(
    adata,
    labeling_time_mask=labeling_time_mask,
    obs_dist_argsort=obs_dist_argsort,
    n_nontrivial_counts=20,
    use_rep="labeled_smoothed",
    n_jobs=N_JOBS,
)

# %%
alpha, gamma, r0, success, opt_res = get_parameters(
    adata=adata,
    use_rep="labeled_smoothed",
    time_key="labeling_time",
    experiment_key="experiment",
    n_neighbors=n_neighbors,
    x0=None,
    n_jobs=N_JOBS,
)

# %%
adata.layers["transcription_rate"] = alpha.loc[adata.obs_names, adata.var_names]
adata.layers["degradation_rate"] = gamma.loc[adata.obs_names, adata.var_names]
adata.layers["r0"] = r0.loc[adata.obs_names, adata.var_names]

# %%
# When initializing the VelocityKernel, specify `vkey="velocity_labeled"`
adata.layers["velocity_labeled"] = (alpha - gamma * adata.layers["labeled_smoothed"]).values
