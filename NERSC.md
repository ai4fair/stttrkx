## Access NERSC Clusters

There are two clusters: _(i)_ Cori (Old), _(ii)_ Perlmutter (New). Before establishing an `ssh` connection, run `./sshproxy.sh` for a passwordless login.

```bash
# create ssh keys
./sshproxy.sh -u aakram
```
This command will create ssh keys, _`nersc, nersc.pub, nersc-cert.pub`_, in the `~/.ssh` directory with 24 Hr validity. Once ssh keys are generated, one can login to a certain cluster as follows:


### _(a) - Login to Cori_

```bash
# Login to Cori
ssh -i ~/.ssh/nersc aakram@cori.nersc.gov
```

### _(b) - Login to Perlmutter_

```bash
# Login to Perlmutter
ssh -i ~/.ssh/nersc aakram@perlmutter-p1.nersc.gov
ssh -i ~/.ssh/nersc aakram@saul-p1.nersc.gov
```

### _(c) - Data Transfer to NERSC_

Use special data nodes for data transfers to NERSC.

```bash
# use data node: dtn01
scp -i ~/.ssh/nersc train_40k.tar.gz aakram@dtn01.nersc.gov:/global/u2/a/aakram/
```

## _Submit Jobs on Cori/Perlmutter_

For interactive run, first use `tmux` to create a sesssion, attach/detach the session as needed. When logging in to **Cori** or **Perlmutter** clusters, one login to a random node. So note this node and `ssh` to that one in order to attach to a `tmux` session whenever needed.

### _1. Interactive Jobs_

There are two ways to allocate resources **interactivly**: _(i)_ `salloc` _(ii)_ `srun --pty bash -l` commands. When using `srun` we span a new bash session using `--pty bash -l`.

* **CPU Resources**

```bash
# activate conda env
conda activate exatrkx-cori
export EXATRKX_DATA=$SCRATCH
```
```bash
# allocate resources (cpu)
salloc -N 1 -q interactive -C haswell -A m3443 -t 04:00:00
  srun -N 1 -q interactive -C haswell -A m3443 -t 04:00:00 --pty /bin/bash -l
```
```bash
# run the pipeline
traintrack configs/pipeline_fulltrain.yaml
```

* **GPU Resources**

```bash
# activate conda env
conda activate exatrkx-cori
export EXATRKX_DATA=$SCRATCH

# load "cgpu" module
module load cgpu
```

```bash
# allocate resources (gpu) interactively
salloc -C gpu -N 1 -G 1 -c 32 -t 4:00:00 -A m3443  # OR
  srun -C gpu -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l
```

```bash
# run the pipeline
traintrack configs/pipeline_fulltrain.yaml
```

* **Exiting**

```bash
# exit
exit

# unload "cgpu" module
module unload cgpu

# deactivate conda env
conda deactivate
```

### _2. Non-interactive Jobs_

For `sbatch` for jobs, two scripts are available: _`submit_cori.sh`_ and _`submit_perlm.sh`_. For **_Cori_**, do the following:

```bash
# load environment
conda activate exatrkx-cori
export EXATRKX_DATA=$CSCRATCH

# load gpu settings (cori)
module load cgpu

# submit job
sbatch submit_cori.sh
```

Alternatively, just run the `submit.jobs` script that will set everything together. The `submit.jobs` looks like the following:

```bash
#!/bin/bash
export SLURM_SUBMIT_DIR=$HOME"/ctd2022"
export SLURM_WORKING_DIR=$HOME"/ctd2022/logs"
mkdir -p $SLURM_WORKING_DIR;

conda activate exatrkx-cori
export EXATRKX_DATA=$CSCRATCH
module load cgpu

sbatch $SLURM_SUBMIT_DIR/submit_cori.sh
```

The same logic applied to the **_Perlmutter_** cluster.
