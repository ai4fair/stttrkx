## Running STT Pipeline

This code uses the _**Exa.TrkX**_ pipeline as a baseline. It uses the `traintrack` to run different stages (_**Processing**_ and _**GNN**_) of the pipeline. This pipeline is intended for **Straw Tube Tracker (STT)** of **PANDA experiment**. STT is part of Central Tracking System (CTS) in the Target Spectrometer of PANDA experiment.

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

### Important Notes: 

- The **Processing** stage can't run within a CUDA enabled envrionment, due to multiprocessing, one needs to run it in CPU-only envrionment. 

- After the **Processing**, one needs to move the data into `train`, `val` and `test` folders by hand as **GNN** stage assumes data distributed in these folders [Maybe in future this will change].

- **GNN** stage will finish with `GNNBuilder` callback, storing the `edge_score` for all events. One re-run this step by using e.g. `traintrack --inference configs/pipeline_quickstart.yaml` but one needs to put `resume_id` in the `pipeline_quickstart`.

- For the rest of inference/prediction steps don't use the `traintrack`. Proceed as follows:
    - Creat a `test_dataloader` from the testset or one can create [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#why-do-i-need-a-datamodule) with `stages=fit, test, predict`. 
    - Load Model from checkpoint
    - Call Trainer with `model` and `test_dataloader`.
    ```bash
    trainer = Trainer()
    dm = MNISTDataModule()
    model = Model()
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.predict(datamodule=dm)
    ```



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
 


