#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

iter=8000
lr=1e-3
wd=1e-4
mmt=0


DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_dog stanford_40)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data SDog120Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 flower102 sdog120 stanford40)


# nohup 
for i in 4
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    FINETUNED_CKPT=result_finetune/resnet50/${DATASET_ABBR}/ckpt.pth

    DIR=result_IPLA
    NAME=resnet50/${DATASET_ABBR}

    nohup python -u finetune.py  --datapath data/${DATASET}/ --iterations ${iter} --dataset ${DATASET_NAME} --name $NAME --batch_size 64 --lr ${lr} --network resnet50 --method IPLA --weight_decay ${wd}  --momentum ${mmt} --output_dir $DIR --checkpoint ${FINETUNED_CKPT} 
done