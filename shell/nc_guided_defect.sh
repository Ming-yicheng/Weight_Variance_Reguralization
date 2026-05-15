

#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)



COVERAGE=strong_coverage
STRATEGY=deepxplore
Model=resnet50
# Method=finetune
# neuron_coverage top_k_coverage strong_coverage
# random deepxplore dlfuzz dlfuzzfirst

for i in 1
# for (( i=0; i<5; i++ ))
do

# for COVERAGE in neuron_coverage top_k_coverage strong_coverage
# do
# for STRATEGY in deepxplore dlfuzz dlfuzzfirst
# do
# for Model in resnet18 resnet50
# do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    # DIR=results/baseline/finetune
    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}

    FINETUNE_DIR=result/result_finetune/${Model}/${DATASET_ABBR}
    RETRAIN_DIR=result/result_retrain/${Model}/${DATASET_ABBR}
    RENOFEATION_DIR=result/result_renofeation/${Model}/${DATASET_ABBR}
    REMOS_DIR=result/result_remos/${Model}/${DATASET_ABBR}
    IPLA_DIR=result/result_IPLA/${Model}/${DATASET_ABBR}
    SEAM_DIR=result/result_seam/${Model}/${DATASET_ABBR}
    WVR_DIR=result/result_wvr_1000/${Model}/${DATASET_ABBR}
    WVR_OUTLIER_DIR=result/result_wvr_1000_outlier/${Model}/${DATASET_ABBR}

    # FINETUNE_DIR=results/res18_ncprune_sum/finetune/resnet18_${DATASET_ABBR}_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1
    # DELTA_DIR=results/res18_ncprune_sum/delta/resnet18_${DATASET_ABBR}_lr1e-2_iter10000_feat5e-1_wd1e-4_mmt0_1
    # WEIGHT_DIR=results/res18_ncprune_sum/weight/resnet18_${DATASET_ABBR}_total0.8_init0.8_per0.1_int10000_lr5e-3_iter10000_feat0_wd1e-4_mmt0_1
    # NCPRUNE_DIR=results/res18_ncprune_sum/ncprune/resnet18_${DATASET_ABBR}_do_total0.1_trainall_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1

    CUDA_VISIBLE_DEVICES=$1 python -u DNNtest/eval_nc_new.py --datapath ./data/${DATASET}/ --dataset ${DATASET_NAME} \
                --name $NAME --network ${Model} \
                --Finetune_ckpt $FINETUNE_DIR/ckpt.pth \
                --Retrain_ckpt $RETRAIN_DIR/ckpt.pth \
                --Renofeation_ckpt $RENOFEATION_DIR/ckpt.pth \
                --ReMoS_ckpt $REMOS_DIR/ckpt.pth \
                --IPLA_ckpt $IPLA_DIR/ckpt.pth \
                --SeaM_ckpt $SEAM_DIR/ckpt.pth \
                --WVR_ckpt $WVR_DIR/ckpt.pth \
                --WVR_outlier_ckpt $WVR_OUTLIER_DIR/ckpt.pth \
                --output_dir results_eval_nc/nc_adv_eval_test/${Model}/${DATASET_ABBR}/${STRATEGY}_${COVERAGE} \
                --batch_size 32 --coverage $COVERAGE --strategy $STRATEGY 

# done
# done
# done
done