#!/bin/bash

############################################################
# Configuration for single node local run

# Set work directory
workdir="/nfs/lambda_stor_01/homes/khippe/github/all-atom-diffusion-transformer"
ulimit -n 64000  # Increase open file limit otherwise torch DDP will fail for some reason

# Add env setup here if needed - locally should source beforehand
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
export OMP_NUM_THREADS=4

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Model variables
latent_dim=8  # 4 / 8
codebook_levels="[8,5,5,5]"

# Logging info
codebook_levels_array=($(echo $codebook_levels | tr -d '[]' | tr ',' ' ')) # Convert string to array
codebook_size=1
for level in "${codebook_levels_array[@]}"; do
    codebook_size=$((codebook_size * level))  # Calculate product
done
latent_str="latent@${latent_dim}"
codebook_str="codebook@${codebook_size}"
name="fsqvae_${latent_str}_codebooksize_${codebook_str}"

# Define the base command with torchrun for multi-GPU training
# torchrun replaces srun for distributed training on a single machine
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
application="python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS}"

dist_options="--master_addr localhost --master_port 12345"

# Application + options
options="$workdir/src/train_autoencoder.py --config-name=train_fsqautoencoder.yaml trainer=ddp logger=wandb name=$name ++autoencoder_module.latent_dim=$latent_dim"

# Combine application and options
CMD="$application $dist_options $options"

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