# Adaptive usage of large biomedical knowledge graph enables accurate and interpretable drug-drug interaction prediction


# Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Data](#Data)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Examples](#Examples)
- [Results](#results)


# Overview

This repository contains the source code of KnowDDI.

# Repo Contents

- [data](./data): the pre-processed dataset of DrugBank and BioSNAP (also known as TWOSIDES).
- [pytorch](./pytorch): the pytorch version code of KnowDDI.
- [raw_data](./raw_data): the origin dataset of DrugBank and BioSNAP.

# Data

We provide the dataset in the [data](data/) folder. 

| Data  | Source | Description
|-------|----------|----------|
| [DrugBank](./pytorch/data/drugbank/) | [This link](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/data/)| A drug-drug interaction network betweeen 1,709 drugs with 136,351 interactions.| 
| [BioSNAP](./pytorch/data/BioSNAP/) | [This link](http://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html)| A drug-drug interaction network betweeen 645 drugs with 46221 interactions.|
| Hetionet | [This link](https://github.com/hetio/hetionet) | The knowledge graph containing 33,765  nodes  out  of  11  types  (e.g.,  gene,  disease,  pathway,molecular function and etc.) with 1,690,693 edges from 23 relation types after preprocessing (To ensure **no information leakage**, we remove all the overlapping edges  between  HetioNet  and  the  dataset).

We provide the mapping file between ids in our pre-processed data and their original name/drugbank id as well as a copy of Hetionet data and their mapping file on [this link](./raw_data).

# System Requirements

## Hardware Requirements

This repository requires only a standard computer with enough RAM to support the in-memory operations. We recommend that your computer contains a GPU.

## Software Requirements

### OS Requirements

The package development version is tested on *Linux*(Ubuntu 18.04) operating systems with CUDA 10.2.

### Python Dependencies
The environment required by the code is as follows.
```
python==3.7.15
pytorch==1.6.0
torchvision==0.7.0
cudatoolkit==10.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
tqdm==4.43.0
dgl-cu102==0.6.1
```

# Installation Guide
Please follow the commands below:
```
cd KnowDDI-codes
conda create -n KnowDDI_pytorch python=3.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install dgl-cu102==0.6.1
pip install -r requirements.txt
cd pytorch
```



# Examples
The default parameters are the best on DrugBank dataset. To train and evaluate the model,you can run the following command.
```
python train.py -e Drugbank
```
Besides, to train and evaluate the model on BioSNAP dataset,you can run the following command.
```
python train.py -e BioSNAP --dataset=BioSNAP --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2
```




# Results
We provide [examples](./pytorch/experiments/) on two datasets with expected experimental results and running times.


