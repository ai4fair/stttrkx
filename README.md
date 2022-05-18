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

- software environment
    - `conda` contains files and instructions for building a conda environment for CPU and GPU

### (1) - Understanding CSV Data

- exploration of raw CSV data
    - `eda` contains notebooks to understand CSV data
- genric code
    - `src` contains generic code for plotting, event building from CSV data

### (2) - Running Pipeline (Recommended)

- Running Stages of the Pipeline in Bash
    - `traintrack configs/pipeline_quickstart.yaml`

### (3) - Running Pipeline, Interactively
To run pipeline stages in Jupyter, use notebooks in the `notebook` folder.


### (4) - Understanding the Pipeline

In order to understand the **output** of each stage, look into these notebooks.

- `stt1_proc.ipynb` for processing
- `stt2_embd.ipynb` for embedding
- `stt3_filter.ipynb` for filtering
- `stt4_gnn.ipynb` for GNN



### (5) - Post Training, the _Track Building_ Stage

After pipeline is finished, the track building and track evaluation is performed on the data from the GNN stage (last stage in the pipeline). Follow the procedure here,

- `trkx_from_gnn.py` runs after GNN stage to build track candidates (help: `trkx_from_gnn.sh`, `trkx_from_gnn.ipynb`)
- `trkx_reco_eval.py` runs after `trkx_from_gnn.py` to evaluate tracks (help: `trkx_reco_eval.sh`, `trkx_reco_eval.ipynb`)


**Inspection on Track Candidates**

Consult following notebooks

- main.ipynb
- eval.ipynb
 


