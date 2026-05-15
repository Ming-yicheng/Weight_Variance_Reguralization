#!/bin/bash
# 这个脚本会从检查点继续训练所有12个模型，
# 并通过从路径中智能推断参数，确保模型结构匹配。

# --- GPU 配置 和 并行任务数控制 ---
export CUDA_VISIBLE_DEVICES=1
MAX_PARALLEL_JOBS=3

# --- 训练参数 ---
NEW_TOTAL_ITERATIONS=40000 
NETWORK="resnet18"
BATCH_SIZE=128

# --- 我们将分三组处理，因为每组的参数推断逻辑不同 ---

# --- 实验一: 固定总数据量 ---
EXP1_PATHS=(
    "result/variance_examples/exp1_fixed_size/resnet18/cifar100_25_classes/ckpt.pth"
    "result/variance_examples/exp1_fixed_size/resnet18/cifar100_50_classes/ckpt.pth"
    "result/variance_examples/exp1_fixed_size/resnet18/cifar100_75_classes/ckpt.pth"
    "result/variance_examples/exp1_fixed_size/resnet18/cifar100_100_classes/ckpt.pth"
)
# 实验一的固定参数
FIXED_TOTAL_SIZE=12500

echo "--- 开始处理 [实验一] 的续训任务 ---"
for ckpt_path in "${EXP1_PATHS[@]}"; do
    # 检查并行任务数
    while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do wait -n; done

    # 从路径中提取类别数 (例如: 从 "..._25_classes/..." 中提取 "25")
    n_cls=$(echo "$ckpt_path" | grep -oP '(?<=cifar100_)[0-9]+(?=_classes)')
    
    echo "启动任务 (Exp1): ${n_cls} classes, fixed total size ${FIXED_TOTAL_SIZE}"
    
    dir_path=$(dirname "${ckpt_path}")
    network_and_name=$(echo "$dir_path" | grep -oP 'resnet18/.*')
    output_dir=$(dirname "${dir_path%/*}")
    LOG_FILE="${dir_path}/resume_training.log"

    nohup python -u classes_variance.py \
        --resume "${ckpt_path}" \
        --iterations ${NEW_TOTAL_ITERATIONS} \
        --name "${network_and_name}" \
        --output_dir "${output_dir}" \
        --network ${NETWORK} \
        --batch_size ${BATCH_SIZE} \
        --num_classes_subset ${n_cls} \
        --fixed_total_size ${FIXED_TOTAL_SIZE} >> "${LOG_FILE}" 2>&1 &
done

# --- 实验二: 固定类别数, 改变数据量 ---
EXP2_PATHS=(
    "result/variance_examples/exp2_data_fraction/resnet18/cifar100_25p_data/ckpt.pth"
    "result/variance_examples/exp2_data_fraction/resnet18/cifar100_50p_data/ckpt.pth"
    "result/variance_examples/exp2_data_fraction/resnet18/cifar100_75p_data/ckpt.pth"
    "result/variance_examples/exp2_data_fraction/resnet18/cifar100_100p_data/ckpt.pth"
)
# 实验二的固定参数
NUM_CLASSES=100

echo "--- 开始处理 [实验二] 的续训任务 ---"
for ckpt_path in "${EXP2_PATHS[@]}"; do
    while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do wait -n; done

    # 从路径中提取数据百分比 (例如: 从 "..._25p_data/..." 中提取 "25")
    frac_percent=$(echo "$ckpt_path" | grep -oP '(?<=cifar100_)[0-9]+(?=p_data)')
    # 将百分比转换为小数 (例如: "25" -> "0.25")
    data_frac=$(echo "scale=2; $frac_percent / 100" | bc)

    echo "启动任务 (Exp2): ${NUM_CLASSES} classes, data fraction ${data_frac}"

    dir_path=$(dirname "${ckpt_path}")
    network_and_name=$(echo "$dir_path" | grep -oP 'resnet18/.*')
    output_dir=$(dirname "${dir_path%/*}")
    LOG_FILE="${dir_path}/resume_training.log"

    nohup python -u classes_variance.py \
        --resume "${ckpt_path}" \
        --iterations ${NEW_TOTAL_ITERATIONS} \
        --name "${network_and_name}" \
        --output_dir "${output_dir}" \
        --network ${NETWORK} \
        --batch_size ${BATCH_SIZE} \
        --num_classes_subset ${NUM_CLASSES} \
        --data_fraction ${data_frac} >> "${LOG_FILE}" 2>&1 &
done

# --- 实验三: 固定每个类的样本量 ---
EXP3_PATHS=(
    "result/variance_examples/exp3_fixed_per_class/resnet18/cifar100_25_classes_500_samples/ckpt.pth"
    "result/variance_examples/exp3_fixed_per_class/resnet18/cifar100_50_classes_500_samples/ckpt.pth"
    "result/variance_examples/exp3_fixed_per_class/resnet18/cifar100_75_classes_500_samples/ckpt.pth"
    "result/variance_examples/exp3_fixed_per_class/resnet18/cifar100_100_classes_500_samples/ckpt.pth"
)
# 实验三的固定参数 (250个样本 = 50%的数据)
DATA_FRACTION=1

echo "--- 开始处理 [实验三] 的续训任务 ---"
for ckpt_path in "${EXP3_PATHS[@]}"; do
    while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do wait -n; done

    # 从路径中提取类别数
    n_cls=$(echo "$ckpt_path" | grep -oP '(?<=cifar100_)[0-9]+(?=_classes)')

    echo "启动任务 (Exp3): ${n_cls} classes, data fraction ${DATA_FRACTION}"

    dir_path=$(dirname "${ckpt_path}")
    network_and_name=$(echo "$dir_path" | grep -oP 'resnet18/.*')
    output_dir=$(dirname "${dir_path%/*}")
    LOG_FILE="${dir_path}/resume_training.log"

    nohup python -u classes_variance.py \
        --resume "${ckpt_path}" \
        --iterations ${NEW_TOTAL_ITERATIONS} \
        --name "${network_and_name}" \
        --output_dir "${output_dir}" \
        --network ${NETWORK} \
        --batch_size ${BATCH_SIZE} \
        --num_classes_subset ${n_cls} \
        --data_fraction ${DATA_FRACTION} >> "${LOG_FILE}" 2>&1 &
done

echo "所有任务均已启动。等待所有任务完成..."
wait
echo "所有续训任务均已成功完成！"