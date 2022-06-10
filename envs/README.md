
## Build Conda Environment

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
&nbsp;
## Build Singularity Container

See the `*.def` files to build a _Singularity_ container with a anaconda environment.

&nbsp;
## Build Shifter Container

Coming soon...

&nbsp;
## How to run STT Pipeline?

```bash
# activate conda env
conda activate exatrkx-cpu

# cd to a Pipeline directory
cd exatrkx-hsf/Pipelines/STT_Example

# export the EXATRKX_DATA variable
export EXATRKX_DATA=$HOME/current/data_sets/exatrkx-hsf

# run TrainTrack by providing a config file
traintrack configs/pipeline_quickstart.yaml

```

### _Stages_

1. **Processing**: _`Pipeline/STT_Example/LightningModules/Processing`_
2. **Embedding**: _`Pipeline/STT_Example/LightningModules/Embedding`_
3. **Filter**: _`Pipeline/STT_Example/LightningModules/Filter`_
4. **GNN**: _`Pipeline/STT_Example/LightningModules/GNN`_

### _Cofiguration of a Stage_

- _`Pipeline/STT_Example/configs`_ contains top-level configuration for **TrainTrack**
- **TrainTrack** get information about different stages of the Pipeline to run the stages in order. For example, the **Processing**, **Embedding**,...,**GNN** stages.
- The **runtime** configuration of each stages lies in a _**config**_ file in respective directory of a stage. For example, for **Processing** stage utilizes config file in _`Pipeline/STT_Example/LightningModules/Processing/configs/prepare_quickstart.yaml`_ to fulfill its requirements.
