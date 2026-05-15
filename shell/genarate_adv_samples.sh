#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

BASE_DATA_PATH="data"
OUTPUT_DIR="adv_samples_early_layer4"

DATASETS=("MIT67Data" "CUB200Data" "Flower102Data" "SDog120Data" "Stanford40Data")
DATASET_PATHS=("MIT_67" "CUB_200_2011" "Flower_102" "stanford_dog" "stanford_40")

SOURCE_MODELS=("resnet18" "resnet50")

ATTACK_METHODS=("pgd" "mim" "tifgsm" "difgsm")

NEURON_TO_ATTACK=0

echo "Starting adversarial sample generation..."

for dataset_idx in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[dataset_idx]}
    DATASET_PATH=${DATASET_PATHS[dataset_idx]}
    
    for model_idx in "${!SOURCE_MODELS[@]}"; do
        SOURCE_MODEL=${SOURCE_MODELS[model_idx]}
        
        for ATTACK_METHOD in "${ATTACK_METHODS[0]}"; do
        
            echo "======================================================================"
            echo "RUNNING TASK:"
            echo "  - Dataset:       $DATASET"
            echo "  - Source Model:  $SOURCE_MODEL"
            echo "  - Attack Method: $ATTACK_METHOD"
            echo "======================================================================"
            
            python adversarial_samples_generation.py \
                --datapath "$BASE_DATA_PATH/$DATASET_PATH" \
                --dataset "$DATASET" \
                --source_model "$SOURCE_MODEL" \
                --attack_method "$ATTACK_METHOD" \
                --neuron_idx "$NEURON_TO_ATTACK" \
                --B 0.1 \
                --m 1000 \
                --attack_iter 40 \
                --batch_size 32 \
                --output_dir "$OUTPUT_DIR"
            
            if [ $? -ne 0 ]; then
                echo "!!!!!!!!!! ERROR !!!!!!!!!!"
                exit 1
            fi
            
        done
    done
done


echo "All tasks completed."