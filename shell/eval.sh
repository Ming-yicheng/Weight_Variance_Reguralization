#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_dog stanford_40)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data SDog120Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 flower102 sdog120 stanford40)

ATTACK_METHOD="IFA"
IFA_LAYER="layer2"
METHODS=(finetune retrain renofeation remos IPLA seam wvr_1000 wvr_1000_outlier)

for dataset_idx in "${!DATASETS[@]}";
do
    network=resnet50
    DATASET=${DATASETS[dataset_idx]}
    DATASET_NAME=${DATASET_NAMES[dataset_idx]}
    DATASET_ABBR=${DATASET_ABBRS[dataset_idx]}

    for CURRENT_METHOD in "${METHODS[@]}"; do
        TRANSFERED_CKPT=result/result_${CURRENT_METHOD}/${network}/${DATASET_ABBR}/ckpt.pth
        if [ "$ATTACK_METHOD" = "IFA" ]; then
            ADV_SAMPLES=adv_samples/${network}/${DATASET_NAME}/${ATTACK_METHOD}/${IFA_LAYER}/eps0.1_iter40_m1000.0_neuron0.pth
            ATTACK_NAME=${ATTACK_METHOD}_${IFA_LAYER}
        else
            ADV_SAMPLES=adv_samples/${network}/${DATASET_NAME}/${ATTACK_METHOD}/eps0.1_iter40_m1000.0_neuron0.pth
            ATTACK_NAME=${ATTACK_METHOD}
        fi

        DIR=results_eval/${CURRENT_METHOD}
        NAME=${network}/${DATASET_ABBR}/${ATTACK_NAME}

        echo "Running: Dataset=${DATASET_ABBR} | Method=${CURRENT_METHOD} | Attack=${ATTACK_METHOD}"

        python -u quick_eval.py \
            --datapath data/${DATASET}/ \
            --dataset ${DATASET_NAME} \
            --name $NAME \
            --network ${network} \
            --output_dir $DIR \
            --checkpoint ${TRANSFERED_CKPT} \
            --adv_samples_path ${ADV_SAMPLES}
    done
done
