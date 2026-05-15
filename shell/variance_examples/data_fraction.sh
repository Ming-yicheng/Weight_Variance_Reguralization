#!/bin/bash
# This script runs Experiment 2:
# Fixes the number of classes and varies the fraction of training data used.

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=1

# --- Experiment Parameters ---
# We use all 100 classes from CIFAR-100
NUM_CLASSES=100
FRACTIONS=(0.25 0.5 0.75 1.0) # The different data fractions to test

# --- Common Hyperparameters ---
ITER=20000          # Iterations can be adjusted for CIFAR-100
LR=0.01
WD=0.0005           # A common weight decay for CIFAR
MMT=0.9
NETWORK="resnet18"
DATASET_PATH="./data"
DATASET_NAME="CIFAR100Data"
BATCH_SIZE=128

echo "Starting Experiment 2: Fixed number of classes = ${NUM_CLASSES}"

# --- Loop and Launch Jobs ---
for frac in ${FRACTIONS[@]}
do
    # Convert fraction to percentage for naming (e.g., 0.25 -> 25)
    frac_percent=$(echo "$frac * 100" | bc | cut -d. -f1)
    echo "--- Launching job for data fraction ${frac_percent}% ---"

    # Define a unique name and output directory for this run
    DIR="result/variance_examples/exp2_data_fraction"
    NAME="${NETWORK}/cifar100_${frac_percent}p_data"
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
        --num_classes_subset ${NUM_CLASSES} \
        --data_fraction ${frac} > "${DIR}/${NAME}.log" 2>&1 &

    echo "Launched ${NAME}. Log will be at ${DIR}/${NAME}.log"
    # sleep 5
done

echo "All jobs for Experiment 2 (Data Fraction) have been launched."
echo "You can monitor progress with: tail -f results/exp2_data_fraction/*/*.log"