# Data can be downloaded from GEO with accession number GSE128365.
# Data needs to be saved in `data/rpe1/raw/`

from functools import reduce

from tqdm import tqdm

import pandas as pd
from scipy.sparse import csr_matrix

from anndata import AnnData

from crp import DATA_DIR

DATASET_ID = "rpe1"
(DATA_DIR / DATASET_ID / "processed").mkdir(parents=True, exist_ok=True)

FILENAME_PREFIX = "GSE128365_SupplementaryData_RPE1_"


if __name__ == "__main__":
    counts = {}

    df = pd.read_csv(DATA_DIR / DATASET_ID / "raw" / f"{FILENAME_PREFIX}labeled_splicedUMI.csv.gz")
    df = df.set_index("Gene_Id").T
    df.columns.name = None

    obs_names = df.index
    var_names = df.columns

    counts["labeled_spliced"] = csr_matrix(df.values)

    for count_type in tqdm(["labeled_unspliced", "unlabeled_spliced", "unlabeled_unspliced"]):
        df = pd.read_csv(DATA_DIR / DATASET_ID / "raw" / f"{FILENAME_PREFIX}{count_type}UMI.csv.gz")
        df = df.set_index("Gene_Id").T
        df.columns.name = None

        df = df.loc[obs_names, var_names].astype(float)
        counts[count_type] = csr_matrix(df.values)

    metadata = pd.read_csv(DATA_DIR / DATASET_ID / "raw" / f"{FILENAME_PREFIX}metadata.csv.gz", index_col=0).T.loc[
        obs_names, :
    ]

    metadata[["experiment", "labeling_time"]] = metadata["Condition_Id"].str.split("_").tolist()

    metadata.columns = metadata.columns.str.lower()
    metadata.columns.name = None
    metadata.rename(
        columns={
            "cell_cycle_possition": "cell_cycle_position",
            "cell_cycle_relativepos": "cell_cycle_rel_position",
            "rfp_log10_corrected": "log10_rfp",
            "gfp_log10_corrected": "log10_gfp",
        },
        inplace=True,
    )

    metadata["well_id"] = metadata["well_id"].astype(float).astype(int)
    metadata["cell_cycle_position"] = metadata["cell_cycle_position"].astype(float).astype(int)

    adata = AnnData(X=reduce(lambda x, y: x + y, counts.values()), layers=counts)
    adata.obs_names = obs_names
    adata.var_names = var_names

    adata.obs = metadata.loc[
        adata.obs_names,
        [
            "plate_id",
            "well_id",
            "labeling_time",
            "experiment",
            "log10_rfp",
            "log10_gfp",
            "cell_cycle_position",
            "cell_cycle_rel_position",
        ],
    ]

    adata.layers["total"] = adata.X.copy()
    adata.layers["labeled"] = adata.layers["labeled_unspliced"] + adata.layers["labeled_spliced"]
    adata.layers["unlabeled"] = adata.layers["unlabeled_unspliced"] + adata.layers["unlabeled_spliced"]

    adata.write_h5ad(DATA_DIR / DATASET_ID / "processed" / "adata.h5ad")
