
Almost every package used in this library is available on conda and has known compatibility for both CPU and GPU (with CUDA v11.3). Therefore installation should be as simple as:

### _1. Build Conda Environment_

**CPU-only Installation:**

```bash
# CPU-only installation
cd envs/
conda env create -f cpu_environment.yml python=3.9
conda activate exatrkx-cpu
pip install -e .
```

**GPU-only Installation:**

```bash
# GPU-only installation
cd envs/
conda env create -f gpu_environment.yml python=3.9
conda activate exatrkx-gpu
pip install -e .
```

**Activate Environment:**

```bash
# e.g. activate exatrkx-cpu
conda activate exatrkx-cpu
```

### _2. Build Container Environment_

See the `*.def` files to build a _Singularity_ container with a anaconda environment.


### _3. Running the Pipeline_

Once a conda environemnt is successfully created, one can run the pipeline from the root directory as follows:

```bash
# running pipeline
conda activate exatrkx-cpu
export EXATRKX_DATA=path/to/dataset
traintrack configs/pipeline_quickstart.yaml
```

#### _3.1 Pipeline Stages_

Top-level pipeline config (_e.g. `configs/pipeline_quickstart.yaml`_) contains the several stages that will run sequentially. One can comment those not needed.

1. **Processing**: _`LightningModules/Processing`_
2. **DNN**: _`LightningModules/DNN`_
3. **GNN**: _`LightningModules/GNN`_
4. **Segmenting**: _`LightningModules/Segmenting`_


#### _3.2 Pipeline Cofiguration_

There are two level of configuration: top-level and stage-level.

- _`configs/`_ contains top-level configuration for the **TrainTrack** 
- _`LightningModules/<Stage Name>/configs/`_ contains stage-level configurations for the **TrainTrack**

