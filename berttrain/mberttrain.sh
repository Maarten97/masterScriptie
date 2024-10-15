#!/bin/bash
#SBATCH -J mBERTtrain                  # Name of the job
#SBATCH -c 1                           # Number of cores
#SBATCH --gres=gpu:1                   # Number of GPU
#SBATCH --mail-type=END,FAIL           # Email status changes
#SBATCH --partition=main,dmb           # Partition
#SBATCH --mem=20GB
#SBATCH --time=4-01:00:00

# Diagnostic information, Display node name
echo "nodename :"
hostname

# Move to local directory
# Define paths
echo "Define Paths"
HOME_DIR=/home/s1722115
SCRATCH_DIR=/local/$SLURM_JOB_ID

TOKEN_DIR=$HOME_DIR/token/token_mbert
MODEL_DIR=$HOME_DIR/model/mbert
CODE_DIR=$HOME_DIR/code/mbert
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

# Move mBERT to local directory
mkdir $SCRATCH_DIR
mkdir $SCRATCH_DIR/mbert
cp "/home/s1722115/model/mbert/config.json" $SCRATCH_DIR/mbert/
cp "/home/s1722115/model/mbert/model.safetensors" $SCRATCH_DIR/mbert/
cp "/home/s1722115/model/mbert/special_tokens_map.json" $SCRATCH_DIR/mbert/
cp "/home/s1722115/model/mbert/tokenizer_config.json" $SCRATCH_DIR/mbert/
cp "/home/s1722115/model/mbert/vocab.txt" $SCRATCH_DIR/mbert/


# Verify the mbert directory was copied
if [ -d "$SCRATCH_DIR/mbert" ]; then
    echo "mbert directory copied successfully."
    ls $SCRATCH_DIR/mbert  # List contents for confirmation
else
    echo "Error: mbert directory not copied!"
    exit 1  # Exit if the copy failed
fi

# Load nvidia cuda toolkit and python
echo "load modules"
ulimit -n 4096
module load nvidia/cuda-12.4
module load python/3.10.7

cp $CODE_DIR/trainmbert1.py $SCRATCH_DIR/
cp $TOKEN_DIR/output_file_1.pt $SCRATCH_DIR/

cd $SCRATCH_DIR

# Run your Python script
echo "Starting run script 1"
srun python3 trainmbert1.py
echo "End script"

cp -r $SCRATCH_DIR/mbert_check1 $OUTPUT_DIR
cp $SCRATCH_DIR/training_log1.txt $OUTPUT_DIR

rm -rf $SCRATCH_DIR/training_log1.txt
rm -rf $SCRATCH_DIR/trainmbert1.py
rm -rf $SCRATCH_DIR/output_file_1.pt

cp $CODE_DIR/trainmbert2.py $SCRATCH_DIR/
cp $TOKEN_DIR/output_file_2.pt $SCRATCH_DIR/

echo "Starting run script 2"
srun python3 trainmbert2.py
echo "End script"

cp -r $SCRATCH_DIR/mbert_check2 $OUTPUT_DIR
cp $SCRATCH_DIR/training_log2.txt $OUTPUT_DIR

rm -rf $SCRATCH_DIR/training_log2.txt
rm -rf $SCRATCH_DIR/trainmbert2.py
rm -rf $SCRATCH_DIR/output_file_2.pt

cp $CODE_DIR/trainmbert3.py $SCRATCH_DIR/
cp $TOKEN_DIR/output_file_3.pt $SCRATCH_DIR/

echo "Starting run script 3"
srun python3 trainmbert3.py
echo "End script"

# Copy everything from scratch to the new directory in your home folder
cp -r $SCRATCH_DIR/* $OUTPUT_DIR/

# Cleanup scratch directory
rm -rf $SCRATCH_DIR

echo "SH script ended fully"