#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

iter=90000
lr=0.01
wd=0.005
mmt=0.9
num_classes_subset=50

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_dog stanford_40)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data SDog120Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 flower102 sdog120 stanford40)

# NUM_GPUS=3
# nohup 
# env CUDA_VISIBLE_DEVICES=${GPU_ID}
for i in 1
do
    # GPU_ID=$((i % NUM_GPUS))

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=result/classes_variance_1
    NAME=resnet18/${DATASET_ABBR}_${num_classes_subset}

    nohup python -u classes_variance.py  --datapath data/${DATASET}/ --iterations ${iter} --dataset ${DATASET_NAME} --name $NAME --batch_size 64 --lr ${lr} --network resnet18 --weight_decay ${wd} --momentum ${mmt} --num_classes_subset ${num_classes_subset} --output_dir ${DIR} 
done