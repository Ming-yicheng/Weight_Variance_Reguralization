#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

iter=30000
lr=1e-3
wd=1e-4
mmt=0
wvr_lambda=1000


DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_dog stanford_40)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data SDog120Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 flower102 sdog120 stanford40)

# NUM_GPUS=3
# nohup 
# env CUDA_VISIBLE_DEVICES=${GPU_ID}
for i in 4
do
    # GPU_ID=$((i % NUM_GPUS))

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=result/result_wvr_${wvr_lambda}
    NAME=resnet18/${DATASET_ABBR}

    # LOG_FILE="${DIR}/${NAME}/training.log"
    # mkdir -p $(dirname ${LOG_FILE}) 

    nohup python -u finetune.py  --datapath data/${DATASET}/ --iterations ${iter} --dataset ${DATASET_NAME} --name $NAME --batch_size 64 --lr ${lr} --network resnet18 --weight_decay ${wd} --method Variance_regularization --momentum ${mmt} --wvr_lambda ${wvr_lambda} --output_dir ${DIR}
done