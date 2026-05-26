# Weight Variance Regularization

This repository contains the code used for experiments with Weight Variance Regularization (WVR) on transfer learning robustness. The current implementation focuses on ResNet backbones and supports layer-wise WVR in both global and outlier modes.

## Scope

Implemented in this repository:

- Fine-tuning and retraining with `finetune.py`
- Weight Variance Regularization (WVR), including `global` and `outlier` modes
- ReMoS-style neuron-coverage-guided pruning workflow
- IPLA
- L2 / weight-decay baseline
- TAA and IFA adversarial sample generation
- Robustness evaluation using pre-generated adversarial samples

Not implemented in this repository:

- `renofeation`
- `seam`

The evaluation scripts may contain `renofeation` and `seam` as checkpoint names for comparison, but their reproduction code is not included here. To evaluate those methods, prepare their checkpoints externally and place them under the expected result directories.

## Environment

The original experiments use a conda environment named `wvr`. For the ResNet + WVR/ReMoS main workflow in this repository, the required Python packages are:

```text
python      3.11.14
torch       2.9.1+cu130
torchvision 0.24.1+cu130
advertorch  0.2.3
numpy       2.3.3
pandas      2.3.3
pillow      11.3.0
scipy       1.16.3
```

Activate the environment with:

```bash
conda activate wvr
```

One possible setup is:

```bash
conda create -n wvr python=3.11 -y
conda activate wvr
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install advertorch==0.2.3 numpy==2.3.3 pandas==2.3.3 pillow==11.3.0 scipy==1.16.3 
```

Notes:

- The training and attack scripts call `.cuda()`, so running full experiments requires a CUDA-capable PyTorch installation and an NVIDIA GPU.
- If your CUDA driver does not support the `cu130` wheel, install the PyTorch/torchvision pair that matches your local CUDA runtime from the official PyTorch index.
- `advertorch==0.2.3` is required for `LinfPGDAttack`, which is used by both evaluation and adversarial sample generation.
- The dataset loaders are project-local and live under `dataset/`.


## Data Layout

Place datasets under `data/` using the directory names expected by the shell scripts:

```text
data/
  MIT_67/
  CUB_200_2011/
  Flower_102/
  stanford_dog/
  stanford_40/
```

The matching dataset class names are:

```text
MIT67Data
CUB200Data
Flower102Data
SDog120Data
Stanford40Data
```

## Supported Backbones

The main training, generation, and evaluation paths currently support ResNet only:

```text
resnet18
resnet34
resnet50
resnet101
```


## Training

All methods are launched through `finetune.py`; the `--method` argument selects the training strategy.

### Fine-tuning and Retraining

Fine-tuning and retraining share the same shell entry:

```bash
bash shell/Finetune.sh
```

They are distinguished by the parameters and output directory you use. For example, a standard fine-tuning run uses the default `--method None` path and writes to a fine-tune result directory. A retraining baseline can use the same script structure with retraining-specific hyperparameters, initialization, and an output directory such as `result/result_retrain`.

### WVR

Use:

```bash
bash shell/WVR.sh
```

Important arguments:

```bash
--method Variance_regularization
--wvr_lambda 1000
--wvr_mode global
```

`--wvr_mode` can be:

- `global`: regularize all convolutional layers
- `outlier`: regularize only high-variance convolutional layers selected from the teacher model

The WVR loss is computed at the layer level.

### ReMoS

Generate coverage profiles first:

```bash
bash shell/remos/nc_profile.sh
```

Then run ReMoS:

```bash
bash shell/remos/remos.sh
```

### IPLA

Use:

```bash
bash shell/IPLA.sh
```

### L2 / Weight Decay Baseline

Use:

```bash
bash shell/l2.sh
```

## Adversarial Sample Generation

Use:

```bash
bash shell/genarate_adv_samples.sh
```

The generator supports one attack method per run:

```bash
--attack_method TAA
```

or:

```bash
--attack_method IFA --ifa_layer layer2
```

For IFA, `--ifa_layer` can be:

```text
layer1
layer2
layer3
layer4
```

Adversarial samples are saved under:

```text
adv_samples/<source_model>/<dataset>/<attack_method>/...
```

For IFA, the selected layer is included:

```text
adv_samples/<source_model>/<dataset>/IFA/<ifa_layer>/...
```

## Evaluation

Use:

```bash
bash shell/eval.sh
```

The script loads checkpoints from:

```text
result/result_<method>/<network>/<dataset_abbr>/ckpt.pth
```

and adversarial samples from:

```text
adv_samples/<network>/<dataset>/<attack_method>/...
```

For methods whose reproduction is not included here, such as `renofeation` and `seam`, place externally generated checkpoints in the expected directories before running evaluation.

## Typical Workflow

1. Prepare datasets under `data/`.
2. Train a baseline with `shell/Finetune.sh`.
3. Train WVR with `shell/WVR.sh`.
4. Optionally train ReMoS, IPLA, or L2 baselines with their corresponding scripts.
5. Generate adversarial samples with `shell/genarate_adv_samples.sh`.
6. Evaluate checkpoints with `shell/eval.sh`.

## Notes

- The shell scripts are templates. Edit GPU ids, dataset loop indices, model names, hyperparameters, and output directories before running full experiments.
- `shell/genarate_adv_samples.sh` keeps the original filename spelling used by the project.
- Large datasets, checkpoints, generated adversarial samples, and result artifacts are not included in this repository.
