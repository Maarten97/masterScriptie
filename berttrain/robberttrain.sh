#!/bin/bash
#SBATCH -J robberttrain                  # Name of the job
#SBATCH -c 1                           # Number of cores
#SBATCH --gres=gpu:1                   # Number of GPU
#SBATCH --mail-type=END,FAIL           # Email status changes
#SBATCH --partition=main,dmb           # Partition
#SBATCH --mem=30GB
#SBATCH --time=1-01:16:00

set -e  # Exit script on any error (non-zero exit code)

# Diagnostic information, Display node name
echo "nodename :"
hostname

# Move to local directory
# Define paths
echo "Define Paths"
HOME_DIR=/home/s1722115
SCRATCH_DIR=/local/$SLURM_JOB_ID

TOKEN_DIR=$HOME_DIR/token/token_robbert
MODEL_DIR=$HOME_DIR/model/robbert
CODE_DIR=$HOME_DIR/code/robbert
OUTPUT_DIR=$HOME_DIR/output/$SLURM_JOB_ID

echo "Create output DIR"
mkdir -p $OUTPUT_DIR

# Log additional information for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : $SLURM_NNODES"
echo "Number of threads           : $SLURM_CPUS_PER_TASK"
echo "Number of threads per core  : $SLURM_THREADS_PER_CORE"
echo "Name of nodes used          : $SLURM_JOB_NODELIST"
echo "Copying files: "

# Move robbert to local directory
mkdir $SCRATCH_DIR
mkdir $SCRATCH_DIR/robbert
cp "/home/s1722115/model/robbert/config.json" $SCRATCH_DIR/robbert/
cp "/home/s1722115/model/robbert/merges.txt" $SCRATCH_DIR/robbert/
cp "/home/s1722115/model/robbert/model.safetensors" $SCRATCH_DIR/robbert/
cp "/home/s1722115/model/robbert/special_tokens_map.json" $SCRATCH_DIR/robbert/
cp "/home/s1722115/model/robbert/tokenizer_config.json" $SCRATCH_DIR/robbert/
cp "/home/s1722115/model/robbert/vocab.json" $SCRATCH_DIR/robbert/

# Verify the robbert directory was copied
if [ -d "$SCRATCH_DIR/robbert" ]; then
    echo "robbert directory copied successfully."
    ls $SCRATCH_DIR/robbert  # List contents for confirmation
else
    echo "Error: robbert directory not copied!"
    exit 1  # Exit if the copy failed
fi

# Load nvidia cuda toolkit and python
echo "load modules"
ulimit -n 4096
module load nvidia/cuda-12.4
module load python/3.10.7

cp $CODE_DIR/trainrobbert1.py $SCRATCH_DIR/
cp $TOKEN_DIR/output_file_1.pt $SCRATCH_DIR/

cd $SCRATCH_DIR

# Loop over the range 1 to 23
for i in $(seq 1 23)
do
  # Define variables for the current iteration
  prev_num=$((i-1))
  next_num=$((i+1))

  # Run Python script
  echo "Starting run script $i"
  python3 $SCRATCH_DIR/trainrobbert$i.py
  echo "End script $i"

  # Copy results to output directory
  mkdir -p $OUTPUT_DIR/robbert_check$i/
  cp -r $SCRATCH_DIR/robbert_check$i/* $OUTPUT_DIR/robbert_check$i/
  cp $SCRATCH_DIR/training_log$i.txt $OUTPUT_DIR

  # Clean up scratch directory
  rm -rf $SCRATCH_DIR/training_log$i.txt
  rm -rf $SCRATCH_DIR/trainrobbert$i.py
  rm -rf $SCRATCH_DIR/output_file_$i.pt

  # Prepare for the next iteration (if applicable)
  if [ $i -lt 23 ]; then
    cp $CODE_DIR/trainrobbert$next_num.py $SCRATCH_DIR/
    cp $TOKEN_DIR/output_file_$next_num.pt $SCRATCH_DIR/
  fi
done

# Copy everything from scratch to the new directory in your home folder
cp -r $SCRATCH_DIR/* $OUTPUT_DIR/

# Cleanup scratch directory
rm -rf $SCRATCH_DIR

echo "SH script ended fully"