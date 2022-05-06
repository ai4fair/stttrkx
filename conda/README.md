
## Install
Thank god that almost every package used in this library is available on conda and has known compatibility for both CPU and GPU (with CUDA v11.3). Therefore installation should be as simple as:

**CPU-only Installation**

```bash
conda env create -f cpu_environment.yml python=3.8
conda activate exatrkx-cpu
pip install -e .
```

Or, after ensuring your GPU drivers are updated to run CUDA v11.3:

**GPU Installation**

```bash
conda env create -f gpu_environment.yml python=3.8
conda activate exatrkx-gpu
pip install -e .
```

## How to run STT Pipeline?

```bash
# activate conda env
conda activate exatrkx-cpu

# cd to Pipeline directory
cd exatrkx-hsf/Pipelines/STT_Example

# export the EXATRKX_DATA
export EXATRKX_DATA=$HOME/current/data_sets/exatrkx-hsf

# run TrainTrack by providing a config file
traintrack configs/pipeline_quickstart.yaml

```

## Stages

- **Processing**: _`Pipeline/STT_Example/LightningModules/Processing`_
- **Embedding**: _`Pipeline/STT_Example/LightningModules/Embedding`_
- **Filter**: _`Pipeline/STT_Example/LightningModules/Filter`_
- **GNN**: _`Pipeline/STT_Example/LightningModules/GNN`_
- etc

## Cofig Files

- `Pipeline/STT_Example/configs` contains top-level configuration for **TrainTrack**
- **TrainTrack** get information about different stages of the Pipeline to run the stages in order. For example, the **Processing**, **Embedding**,... stages.
- The **runtime** configuration of each stages lies in a `config` file in respective directory such as `Pipeline/STT_Example/LightningModules/<Stage>`.
