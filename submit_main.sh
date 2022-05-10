#!/bin/bash

# *** Account, etc ***
#SBATCH -A panda                                             # Account Name (--account=g2020014)
#SBATCH -J CTD22                                             # Job Name (--job-name=HitPairs)
#SBATCH -M virgo                                             # Cluster Name (--clusters=snowy)
#SBATCH -t 08:00:00                                          # Time (DD-HH:MM) (--time=HH:MM:SS)

# *** Resources, etc ***
#SBATCH -p main                                              # Requested Partition on Cluster (--partition=<name>)
#SBATCH -N 1                                                 # No. of Compute Nodes Requested (--nodes=<number>)
#SBATCH -c 32                                                # No. of CPUs Cores per Task (--cpus-per-task=<ncpus>
#SBATCH --mem=32768                                          # Request CPU RAM, I am requesting 32G
#SBATCH --ntasks-per-node=1                                  # Request no. of tasks invoked on each node (for --nodes)
#SBATCH --gres=gpu:tesla:1                                   # Request GPUs resources [gpu:<type>:<no.>]
#-SBATCH --nodelist=lxbk0[717-718]                           # Request specific node(s) on --partition

# *** I/O ***
#SBATCH -D .                                                 # Set CWD as Wroking Dir. for Batch Script(or to $LUSTRE_HOME)
#SBATCH -o logs/%x-%j.out                                    # Standard Output (--output=<filename pattern>)
#SBATCH -e logs/%x-%j.err                                    # Standard Error (--error=<filename pattern>)
#SBATCH --mail-type=END                                      # Notification Type
#SBATCH --mail-user=a.akram@gsi.de                           # Email for notification

echo "== --------------------------------------------"
echo "== Starting Run at $(date)"                            # Print current data
echo "== SLURM Cluster: ${SLURM_CLUSTER_NAME}"               # Print cluster name (if -M <cluster-name> is used)
echo "== --------------------------------------------"
echo "== SLURM CPUS on GPU: ${SLURM_CPUS_PER_GPU}"           # Only set if the --cpus-per-gpu is specified.
echo "== SLURM CPUS on NODE: ${SLURM_CPUS_ON_NODE}"          #
echo "== SLURM CPUS per TASK: ${SLURM_CPUS_PER_TASK}"        # Only set if the --cpus-per-task is specified.
echo "== --------------------------------------------"
echo "== SLURM No. of GPUS: ${SLURM_GPUS}"                   # Only set if the -G, --gpus option is specified.
echo "== SLURM GPUS per NODE: ${SLURM_GPUS_PER_NODE}"        #
echo "== SLURM GPUS per TASK: ${SLURM_GPUS_PER_TASK}"        #
echo "== --------------------------------------------"
echo "== SLURM Job ID: ${SLURM_JOB_ID}"                      # OR SLURM_JOBID. The ID of the job allocation.
echo "== SLURM Job ACC: ${SLURM_JOB_ACCOUNT}"                # Account name associated of the job allocation.
echo "== SLURM Job NAME: ${SLURM_JOB_NAME}"                  # Name of the job.
echo "== SLURM Node LIST: ${SLURM_JOB_NODELIST}"             # OR SLURM_NODELIST. List of nodes allocated to job.
echo "== SLURM No. of NODES: ${SLURM_JOB_NUM_NODES}"         # OR SLURM_NNODES. Total #nodes in job's resources.
echo "== SLURM No. of CPUs/NODE: ${SLURM_JOB_CPUS_PER_NODE}" #
echo "== --------------------------------------------"
echo "== SLURM Node ID: ${SLURM_NODEID}"                     # ID of the nodes allocated.
echo "== SLURM Node Name: ${SLURMD_NODENAME}"                # Name of the node running the job script
echo "== SLURM No. of Tasks: ${SLURM_NTASKS}"                # OR SLURM_NPROCS. Similar as -n, --ntasks
echo "== SLURM No. of Tasks/Core: ${SLURM_NTASKS_PER_CORE}"  # Only set if the --ntasks-per-core is specified.
echo "== SLURM No. of Tasks/Node: ${SLURM_NTASKS_PER_NODE}"  # Only set if the --ntasks-per-node is specified.
echo "== SLURM Submit Dir. : ${SLURM_SUBMIT_DIR}"            # Dir. where sbatch was invoked. Flag: -D, --chdir.
echo "== --------------------------------------------"

export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
export EXATRKX_DATA=$LUSTRE_HOME/ctd2022
echo "EXATRKX_DATA: $EXATRKX_DATA"

CENV=exatrkx
CONT=gpu_stttrkx.sif
singularity run --nv $LUSTRE_HOME/containers/$CONT -c "conda activate $CENV && traintrack configs/pipeline_quickstart.yaml"
