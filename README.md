## Running CTD2022 Pipeline

This code uses the Exa.TrkX pipeline as a baseline. It uses the `traintrack` to run different stages (_**Processing**_ and _**GNN**_) of the pipeline.

&nbsp;

To successfully run the pipeline, follow these steps.

* Build a `conda` env using files provided in **conda** dir.

```bash
# CPU-only installation
conda env create -f cpu_environment.yml python=3.9
conda activate exatrkx-cpu
pip install -e .
```

```bash
# GPU-only installation
conda env create -f gpu_environment.yml python=3.9
conda activate exatrkx-cpu
pip install -e .
```

* Provide `path` to dataset

```bash
# EXATRKX_DATA is required
export EXATRKX_DATA=path/to/dataset
```

* Activate a `conda` environment and run the pipeline

```bash
# e.g. activate exatrkx-cpu
conda activate exatrkx-cpu
traintrack configs/pipeline_quickstart.yaml
```

&nbsp;

Important Notes: 

- The **Processing** stage can't run within a CUDA enabled envrionment, due to multiprocessing, one needs to run it in CPU-only envrionment. 

- After the **Processing**, one needs to move the data into `train`, `val` and `test` folders by hand as **GNN** stage assumes data distributed in these folders [Maybe in future this will change].

&nbsp;

## Running Pipeline on NERSC

Follow instructions on [NERSC Documentation](https://docs.nersc.gov/) or see the concise and essential version in `NERSC.md`.

&nbsp;

## Directory Tree

&nbsp;
### (1) - Software Environment

- `conda/` contains files and instructions for building a conda environment for CPU and GPU

&nbsp;
### (2) - Understanding CSV Data

- exploration of raw CSV data
    - `eda/` contains notebooks to understand CSV data, any other investigative stuff.
- helper code for raw CSV data
    - `src/` contains helper code for **utility functions**, **plotting**, **event building**, etc

&nbsp;
### (3) - Running Pipeline

- Running Stages of the Pipeline in Bash (Recommended)
    - `traintrack configs/pipeline_quickstart.yaml`

- Running Stages of the Pipeline in Jupyter
    - use notebooks in the `notebook/` folder.

**Note:** Deep learning pipeline specific code rests in `configs/` and `LightningModules/` folders. These two folders are only required to fully run the pipeline. Everything else, is helper code used for post-training analysis. 

&nbsp;
### (4) - Understanding the Pipeline

In order to understand the **output** of each stage, look into these notebooks.

- `stt1_proc.ipynb` for processing stage
- `stt2_embd.ipynb` for embedding stage
- `stt3_filter.ipynb` for filtering stage
- `stt4_gnn.ipynb` for GNN stage
- `main.ipynb` for random testing such as reading CSVs, event building, plotting, etc.


&nbsp;
### (5) - Post Training

After pipeline is finished, all we get is the `edge_score`. One needs to use this information to build tracks. All post-training hepler code is located in the `eval/` directory.

The track building and track evaluation is performed on the data from the GNN stage (last stage in the pipeline). Follow the procedure here,

1. First, run `trkx_from_gnn.py`
2. Second, run `trkx_reco_eval.py`

One can use the corresponding `bash` scripts to facilitate the execution of `python` scripts.

&nbsp;

### (6) - Inspection on Track Candidates

To understand the evaluation of tracks, one needs to dive deeper into the track reconstruction and evalution. Consult the following notebook:

- `eval.ipynb`: Inspections on Track Evaluation.
 


