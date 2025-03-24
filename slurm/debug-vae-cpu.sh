#!/bin/bash

############################################################
# Configuration for single node local run

# Set work directory
workdir="/nfs/lambda_stor_01/homes/khippe/github/all-atom-diffusion-transformer"
ulimit -n 64000  # Increase open file limit for large datasets

# Add env setup here if needed - locally should source beforehand
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

# Specify which GPUs to use
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1
# Model variables
latent_dim=8  # 4 / 8
loss_kl=0.00001  # 0.0001 / 0.00001

# Logging info
latent_str="latent@${latent_dim}"
kl_str="kl@${loss_kl}"
name="vae_${latent_str}_${kl_str}"

# Define the base command with torchrun for multi-GPU training
# torchrun replaces srun for distributed training on a single machine
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
application="python src/train_autoencoder.py"

# Application + options
options="trainer=cpu logger=wandb name=$name ++autoencoder_module.latent_dim=$latent_dim ++autoencoder_module.loss_weights.loss_kl.mp20=$loss_kl ++autoencoder_module.loss_weights.loss_kl.qm9=$loss_kl"

# Combine application and options
CMD="$application $options"

############################################################
# Execution

cd $workdir
echo -e "Changed directory to `pwd -P`.\n"

echo "Time: `date`"
echo "Running on: `hostname`"
echo "Current directory: `pwd -P`"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"

echo -e "\nExecuting command:\n==================\n$CMD\n"

# Create logs directory if it doesn't exist
mkdir -p logs

# Execute the command and redirect output to log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# eval $CMD > logs/train_${TIMESTAMP}.out 2> logs/train_${TIMESTAMP}.err
eval $CMD