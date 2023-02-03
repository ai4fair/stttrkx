## Hyperparameter Tuning: RayTune

RayTune is used to optimize hyperparameters for neural networks. Ray Tune has been tested for DNN stage (see `tune_base, tune_network, tune_quickstart.yaml`).

&nbsp;

To successfully run RayTune, use `tune_quickstart.yaml` config file. RayTune failed with following options:

- **layernorm**: `True` (turn `False`)
- **batchnorm**: `True` (turn `False`)

In the `make_mlp()`, these options additional layers `nn.LayerNorm` and `nn.BatchNorm1d`. Using these layers gives an `error` with current settings in **ASHASchedular** (`TuneASH.py`) and **PBTSchedular** (`TunePBT.py`). 


### _1. RayTune: Tuning_

```bash
# ASHA Schedular
conda activate exatrkx-cpu
export EXATRKX_DATA=$PWD
python TuneASH.py configs/tune_quickstart.py
```

```bash
# PBT Schedular
conda activate exatrkx-cpu
export EXATRKX_DATA=$PWD
python TunePBT.py configs/tune_quickstart.py
```

### _2. RayTune: Analysis_
```bash
# RayTune: Analysis in Tune.ipynb
conda activate exatrkx-cpu
export EXATRKX_DATA=$PWD
juypyter lab
Tune.ipynb
```
