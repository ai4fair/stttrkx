
## _1. Running the Pipeline_

This code uses the _`Exa.TrkX-HSF`_ pipeline as a baseline. It uses the _`traintrack`_ library to run different stages (_Processing_, _DNN/GNN_, and _Segmenting_) of the STT pipeline. This pipeline is intended for the **Straw Tube Tracker (STT)** of the PANDA experiment which is part of the Central Tracking System (CTS) located in the Target Spectrometer of the PANDA experiment.



### _1.1 Running the Pipeline on CPU_

Once a conda environment is successfully created (see _`envs/README.md`_ for building a conda environment), one can run the pipeline from the root directory as follows:

```bash
# running pipeline
conda activate exatrkx-cpu
export EXATRKX_DATA=path/to/dataset
traintrack configs/pipeline_quickstart.yaml
```

### _1.2 Running Pipeline on Cluster_

Follow instructions on [NERSC Documentation](https://docs.nersc.gov/) or see the concise and essential version in _`NERSC.md`_ to run pipeline on the Cori cluster at NERSC. 


## _2. Understanding the Pipeline_

 - **Processing** stage can't run within a CUDA enabled envrionment, due to `multiprocessing` python library, one needs to run it in CPU-only envrionment. After this stage one needs to distribute data into `train`, `val` and `test` folders by hand as **GNN** stage assumes data distributed in these folders [Maybe in future this will change].

- **DNN/GNN** stage will finish with `GNNBuilder` callback, storing the `edge_score` for all events. One can re-run this step by using e.g. `traintrack --inference configs/pipeline_quickstart.yaml` but one needs to put `resume_id` in the `pipeline_quickstart`.

- **Segmentig** stage is meant for track building using DBSCAN or CCL. However, one may skip this stage altogether and move to `eval/` folder where one can perform segmenting as well as track evaluation. This is due to post analysis needs, as one may need to run segmenting together with evaluation using different settings. At the moment, it is recommended to skip this stage and directly move to `eval/` directory (see `eval/README.md` for more details).


## _3. Code Explanation_

The _`stttrkx`_ repo contains several subdirectories containing code for specific tasks. The detail of these subdirectories is as follows:

- _`configs/`_ contains top-level pipeline configuration files for `traintrack`
- _`eda/`_ contains notebooks for **exploratory data analysis** to understand raw data.
- _`envs/`_ contains files for building a **conda environment**
- _`eval/`_ contains code for **track evalution**, however, it also contain code for running **segmenting** stage independently of `traintrack`
- _`LightningModules/`_ contains code for each stage
- _`src/`_ contains helper code for **utility functions**, **plotting**, **event building**, etc
- _`RayTune/`_ contains helper code for running hyperparameter tuning using **Ray Tune** library


Several notebooks are avaialble to inpect output of each stage as well as for post analysis.