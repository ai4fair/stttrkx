#!/bin/sh

LUSTRE_HOME="/lustre/pbar/"$USER

# *** Account, ***
#SBATCH -A aakram                                            # Account Name (--account=g2020014)
#SBATCH -J CTD                                               # Job Name (--job-name=HitPairs)
#-SBATCH -M snowy                                            # Cluster Name (--clusters=snowy)
#SBATCH -t 8:00:00                                           # Time (DD-HH:MM) (--time=0:59:00)
#SBATCH -p main                                              # Partition (node/core/devcore) (--partition=node)
#SBATCH -N 5                                                 # No. of Nodes Requested (--nodes=5)


# *** Resources, etc ***
#SBATCH --exclusive                                          # Don't share nodes with other running jobs
#SBATCH --cpus-per-task=16                                   # CPUs/Task (-c, --cpus-per-task=<ncpus>)
#SBATCH --gres=gpu:v100:1                                    # Request GPUs resources
#-SBATCH --qos=short                                         # Priority of short-jobs of 1-4 nodes, with timelimit<= 15 min.

# *** I/O ***
#SBATCH -D .                                                 # Set CWD as Wroking Dir. for Batch Script(one can also set to $LUSTRE_HOME)
#SBATCH -o %x-%j.out                                         # Standard Output (--output=<filename pattern>)
#SBATCH -e %x-%j.err                                         # Standard Error (--error=<filename pattern>)
#SBATCH --mail-type=END                                      # Notification Type
#SBATCH --mail-user=a.akram@gsi.de                           # Email for notification

echo "== --------------------------------------------"
echo "== Starting Run at $(date)"                            #
echo "== SLURM Cluster: ${SLURM_CLUSTER_NAME}"               #
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



CENV=${exatrkx}
CONT=${gpu_stttrkx.sif}
srun singularity run --nv $LUSTRE_HOME/containers/$CONT -c "conda activate $CENV && python main.py"

