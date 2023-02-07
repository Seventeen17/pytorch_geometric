

# pytorch version limit
>=1.4.0, which is pytorch nighly/master  or pytorch2.0
cuda version: 11.7 or 11.8
install pytorch nightly:

```shell
 conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
```

# install pyg engine libraies

ensure that cudnn version is comparable to cuda version

```
pip install git+https://github.com/pyg-team/pyg-lib.git
pip install git+https://github.com/rusty1s/pytorch_cluster.git
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install git+https://github.com/rusty1s/pytorch_sparse.git
```

# install pyg

```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

