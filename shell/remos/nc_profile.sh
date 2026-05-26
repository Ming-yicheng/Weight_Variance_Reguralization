export PYTHONPATH=../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)

# coverages: neuron_coverage top_k_coverage strong_coverage
# strategies: random deepxplore dlfuzz dlfuzzfirst

for MODEL in resnet18
do
for dataset_idx in 1
do
for COVERAGE in neuron_coverage 
do
for STRATEGY in deepxplore 
do


    DATASET=${DATASETS[dataset_idx]}
    DATASET_NAME=${DATASET_NAMES[dataset_idx]}
    DATASET_ABBR=${DATASET_ABBRS[dataset_idx]}
    NAME=${MODEL}_${DATASET_ABBR}

    python -u nc_prune/my_profile.py --datapath data/${DATASET}/ --dataset ${DATASET_NAME} --name $NAME --network $MODEL --output_dir result/nc_profiling/${COVERAGE}_${DATASET_ABBR}_${MODEL} --batch_size 32 --coverage $COVERAGE --strategy $STRATEGY 

done
done
done
done
