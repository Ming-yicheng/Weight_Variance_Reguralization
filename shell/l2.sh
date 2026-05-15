#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

iter=30000
lr=1e-3
wd=0.1
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_dog stanford_40)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data SDog120Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 flower102 sdog120 stanford40)

# nohup 
for i in 0
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=result_l2
    NAME=resnet18/${DATASET_ABBR}

    nohup python -u finetune.py  --datapath data/${DATASET}/ --iterations ${iter} --dataset ${DATASET_NAME} --name $NAME --batch_size 64 --lr ${lr} --network resnet18 --weight_decay ${wd}  --momentum ${mmt} --output_dir $DIR  
done