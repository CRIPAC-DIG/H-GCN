# H-GCN
## Description
This is the repository for the IJCAI-19 paper [Hierarchical Graph Convolutional Networks for Semi-supervised Node Classiﬁcation](https://arxiv.org/pdf/1902.06667.pdf).

## Requirements

- Tensorflow (1.9.0)
- networkx

## Usage

You can conduct node classification experiments on citation network (Cora, Citeseer or Pubmed) with the following commands:

```bash
python train.py --dataset cora --epochs 60 --early_stopping 1000 --coarsen_level 4 --dropout 0.85 --weight_decay 7e-4 --hidden 32 --node_wgt_embed_dim 8 --seed1 156 --seed2 136
```

```bash
python train.py --dataset citeseer --epochs 200 --early_stopping 60 --coarsen_level 4 --dropout 0.85 --weight_decay 7e-4 --hidden 30 --node_wgt_embed_dim 15 --seed1 156 --seed2 156
```

```bash
python train.py --dataset pubmed --epochs 250 --early_stopping 1000 --coarsen_level 4 --dropout 0.85 --weight_decay 7e-4 --hidden 30 --node_wgt_embed_dim 8 --seed1 156 --seed2 136
```

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{hgcn_ijcai19,
    title = {Hierarchical Graph Convolutional Networks for Semi-supervised Node Classification},
    author = {Fenyu Hu and Yanqiao Zhu and Shu Wu and Liang Wang and Tieniu Tan},
    booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, (IJCAI)},
    year = {2019},
    url = {https://arxiv.org/abs/1902.06667}
}
```