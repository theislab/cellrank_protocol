# %% [markdown]
# # RNA velocity in spermatogenesis
#
# RNA velocity analysis with the *VI model* using data preprocessed with `velocyto`.

# %% [markdown]
# **Requires**
#
# * `adata_generation.ipynb`
#
# **Output**
#
# * `velocyto_var_names.csv`
# * `DATA_DIR/spermatogenesis/velocities/velocyto_velovi.npy`

# %% [markdown]
# ## Library imports

# %%
import anndata as ad
import scanpy as sc
import scvelo as scv

from crp.core import DATA_DIR

# %%
sc.logging.print_version_and_date()

# %% [markdown]
# ## General settings

# %%
# set verbosity levels
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")
scv.settings.plot_prefix = ""

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / "spermatogenesis" / "processed").mkdir(exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / "spermatogenesis" / "raw" / "velocyto.h5ad")
adata

# %% [markdown]
# ## Data pre-processing

# %%
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=False)
sc.pp.log1p(adata)

adata

# %%
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

# %%
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %% [markdown]
# ## Model fitting

# %%
scv.tl.recover_dynamics(adata, var_names="all", n_jobs=10)
scv.tl.velocity(adata, mode="dynamical")

# %% [markdown]
# ### Save results

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / "spermatogenesis" / "processed" / "adata.h5ad")
