#!/bin/bash
#SBATCH -J trainGPUtest                  # Name of the job
#SBATCH -c 2                           # Number of cores
#SBATCH --gres=gpu:2                          # Number of cores
#SBATCH --mail-type=END,FAIL     # Email status changes
#SBATCH --sockets-per-node=1
#SBATCH --partition=main,dmb  # Partition
#SBATCH --mem=16GB
#SBATCH --time=1-01:00:00

# Diagnostic information
# Display node name
echo "nodename :"
hostname

# Load nvidia cuda toolkit and python
echo "load modules"
ulimit -n 4096
module load nvidia/cuda-12.4
module load python/3.10.7

# Move to local directory
# Define paths
HOME_DIR=/home/s1722115/mbert
SCRATCH_DIR=/local/$SLURM_JOB_ID

# Create local scratch directory
mkdir -p $SCRATCH_DIR

# Copy Python script and txt file to scratch
cp $HOME_DIR/pretraintwo.py $SCRATCH_DIR/
cp $HOME_DIR/312997/tokenized_chunks/tokenized_chunk_0.pt $SCRATCH_DIR/
cp -r $HOME_DIR/mbert $SCRATCH_DIR/

# Change to scratch directory
cd $SCRATCH_DIR

# Log additional information for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : $SLURM_NNODES"
echo "Number of threads           : $SLURM_CPUS_PER_TASK"
echo "Number of threads per core  : $SLURM_THREADS_PER_CORE"
echo "Name of nodes used          : $SLURM_JOB_NODELIST"
echo "Starting worker: "

# Run your Python script
srun python3 pretraintwo.py

echo "Exit script"
# Create a directory in your home folder to store the output
OUTPUT_DIR=$HOME_DIR/$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR

# Copy everything from scratch to the new directory in your home folder
cp -r $SCRATCH_DIR/* $OUTPUT_DIR/

# Cleanup scratch directory
rm -rf $SCRATCH_DIR
