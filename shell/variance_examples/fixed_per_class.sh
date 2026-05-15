#!/bin/bash
# This script runs Experiment 3:
# Fixes the number of samples PER CLASS and varies the number of classes.

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=0

# --- Experiment Parameters ---
# Let's fix the samples per class to 250, which is 50% of the available 500.
# You can change this fraction to run with a different number of samples per class.
SAMPLES_PER_CLASS_FRACTION=1
CLASS_SUBSETS=(25 50 75 100) # The different class counts to test

# --- Common Hyperparameters ---
ITER=20000
LR=0.01
WD=0.0005
MMT=0.9
NETWORK="resnet18"
DATASET_PATH="./data"
DATASET_NAME="CIFAR100Data"
BATCH_SIZE=128

SAMPLES_PER_CLASS=$(echo "500 * ${SAMPLES_PER_CLASS_FRACTION}" | bc | cut -d. -f1)
echo "Starting Experiment 3: Fixed samples per class = ${SAMPLES_PER_CLASS}"

# --- Loop and Launch Jobs ---
for n_cls in 100
do
    echo "--- Launching job for ${n_cls} classes ---"

    # Define a unique name and output directory for this run
    DIR="result/variance_examples/exp3_fixed_per_class"
    NAME="${NETWORK}/cifar100_${n_cls}_classes_${SAMPLES_PER_CLASS}_samples"
    LOG_FILE="${DIR}/${NAME}.log"

    # Create the directory for the log file before running the command
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
        --data_fraction ${SAMPLES_PER_CLASS_FRACTION} > "${LOG_FILE}" 2>&1
        
    echo "Launched ${NAME}. Log will be at ${LOG_FILE}"
done

echo "All jobs for Experiment 3 (Fixed Samples Per Class) have been launched."
echo "You can monitor progress with: tail -f results/exp3_fixed_per_class/*/*.log"