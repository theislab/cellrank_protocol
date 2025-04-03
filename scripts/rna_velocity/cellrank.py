# %% [markdown]
# # Application of the VelocityKernel
#
# In this analysis, we infer cell-cell transition probabilities using the VelocityKernel.

# %% [markdown]
# ## Library imports

# %%
import anndata as ad
import cellrank as cr

from crp import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
cr.settings.verbosity = 4

# %% [markdown]
# ## Constants

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "velocitykernel").mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "svg"

# %%
SAVE_RESULTS = True
if SAVE_RESULTS:
    (DATA_DIR / "spermatogenesis" / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / "spermatogenesis" / "processed" / "adata.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# ## VelocityKernel

# %% [markdown]
# ### Kernel

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()

# %%
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
combined_kernel = 0.8 * vk + 0.2 * ck
