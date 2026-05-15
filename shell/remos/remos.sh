export CUDA_VISIBLE_DEVICES=1

iter=30000
lr=1e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)


for i in 1
do
    for ratio in 0.1
    do
        for COVERAGE in neuron_coverage
        do

        total_ratio=${ratio}

        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        DATASET_ABBR=${DATASET_ABBRS[i]}

        NAME=mobilenet/${DATASET_ABBR}
        DIR=result/result_remos

        nohup python finetune.py --iterations ${iter} --datapath data/${DATASET}/ --dataset ${DATASET_NAME} --name ${NAME} --batch_size 64 --lr ${lr} --network mobilenet_v2 --weight_decay ${wd} --momentum ${mmt} --output_dir ${DIR} --method remos --weight_total_ratio $total_ratio --weight_init_prune_ratio $total_ratio --prune_interval $iter --weight_ratio_per_prune 0 --nc_info_dir result/nc_profiling/${COVERAGE}_${DATASET_ABBR}_mobilenet_v2

        done
    done
done