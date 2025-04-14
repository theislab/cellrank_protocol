# CellRank protocols

This repository provides the protocols and corresponding preprocessing for CellRank-based trajectory inference. Data
used in the different use cases is made available in [this](https://doi.org/10.6084/m9.figshare.c.7752290.v1)
figshare collection.

## Installation

To install the software required for running the protocols, we suggest the following workflow

```bash
conda create -n crp-py311 python=3.11 --yes && conda activate crp-py311
conda install -c conda-forge cellrank

git clone https://github.com/theislab/cellrank_protocol.git
cd cellrank_protocol
pip install -e ".[jupyter]"

python -m ipykernel install --user --name crp-py311 --display-name "crp-py311"
```
