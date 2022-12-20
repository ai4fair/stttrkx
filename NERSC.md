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

## Submit Jobs on Cori

Use sbatch for job submission. For example, _`$ sbatch submit_cori.sh`_. However, one can work interactively using either `salloc` or `srun`.


In case of interactive run, use `tmux` and detach the session as needed. One Cori, one can login to a random node. Note the node and `ssh` to that to attach to `tmux` session.

* **CPU Resources**

```bash
# activate conda env
conda activate exatrkx-cori
export EXATRKX_DATA=$SCRATCH

# allocate resources (cpu)
salloc -N 1 -q interactive -C haswell -A m3443 -t 04:00:00

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


There are two ways to allocate resources interactivly: _(i)_ `salloc` _(ii)_ `srun --pty bash -l`

```bash
# allocate resources (gpu) interactively
salloc -C gpu -N 1 -G 1 -c 32 -t 4:00:00 -A m3443  # OR
  srun -C gpu -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l

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



