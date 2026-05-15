#!/bin/bash
# This script runs Experiment 1:
# Fixes the total number of training samples and varies the number of classes.

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=0

# --- Experiment Parameters ---
# According to our design, the total size is based on 25 classes * 500 images/class
FIXED_TOTAL_SIZE=12500
CLASS_SUBSETS=(25 50 75 100) # The different class counts to test

# --- Common Hyperparameters ---
ITER=20000          # Iterations can be adjusted for CIFAR-100
LR=0.01
WD=0.0005           # A common weight decay for CIFAR
MMT=0.9
NETWORK="resnet18"
DATASET_PATH="./data" # Assumes data is in a 'data' subdirectory
DATASET_NAME="CIFAR100Data"
BATCH_SIZE=128

echo "Starting Experiment 1: Fixed total training size = ${FIXED_TOTAL_SIZE}"

# --- Loop and Launch Jobs ---
for n_cls in 100
do
    echo "--- Launching job for ${n_cls} classes ---"

    # Define a unique name and output directory for this run
    DIR="result/variance_examples/exp1_fixed_size"
    NAME="${NETWORK}/cifar100_${n_cls}_classes"
    LOG_FILE="${DIR}/${NAME}.log"

    # --- FIX: Create the directory for the log file BEFORE running the command ---
    mkdir -p $(dirname ${LOG_FILE})

    # Use nohup to run the training in the background
    nohup python -u classes_variance.py \
        --datapath ${DATASET_PATH} \
        --dataset ${DATASET_NAME} \
        --iterations ${ITER} \
        --network ${NETWORK} \
        --name "${NAME}" \
        --output_dir ${DIR} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight_decay ${WD} \
        --momentum ${MMT} \
        --num_classes_subset ${n_cls} \
        --fixed_total_size ${FIXED_TOTAL_SIZE} > "${DIR}/${NAME}.log" 2>&1
        
    echo "Launched ${NAME}. Log will be at ${DIR}/${NAME}.log"
    # Optional: add a sleep if you are launching many jobs and want to avoid disk/cpu contention at startup
    # sleep 5 
done

echo "All jobs for Experiment 1 (Fixed Size) have been launched."
echo "You can monitor progress with: tail -f results/exp1_fixed_size/*/*.log"