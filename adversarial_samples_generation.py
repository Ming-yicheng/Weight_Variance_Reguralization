import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F

from torchvision import transforms
from advertorch.attacks import LinfPGDAttack

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101


def taa_loss(yhat, y):
    beta = 0.1
    weights = torch.ones_like(yhat)
    weights[y == 0] = beta
    return -(((yhat - y) ** 2) * weights).mean()


class FeatureDestructionLoss:
    def __init__(self, model, source_model_name, ifa_layer):
        self.model = model
        self.target_features = None
        self.current_features = None
        self.layer = resolve_ifa_layer(model, source_model_name, ifa_layer)
        self.handle = self.layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.current_features = output

    def set_clean_reference(self, x):
        with torch.no_grad():
            self.model(x)
            self.target_features = self.current_features.detach().clone()

    def __call__(self, yhat, y):
        if self.target_features is None:
            return torch.tensor(0.0, device=yhat.device)

        adv_flat = self.current_features.reshape(self.current_features.size(0), -1)
        clean_flat = self.target_features.reshape(self.target_features.size(0), -1)
        return -F.cosine_similarity(adv_flat, clean_flat, dim=1).mean()

    def remove(self):
        self.handle.remove()


def normalize_attack_method(method):
    method = method.upper()
    if method not in {"TAA", "IFA"}:
        raise ValueError(f"Unsupported attack method: {method}")
    return method


def resolve_ifa_layer(model, source_model_name, ifa_layer):
    if "resnet" in source_model_name:
        if ifa_layer not in {"layer1", "layer2", "layer3", "layer4"}:
            raise ValueError("ResNet IFA layer must be one of: layer1, layer2, layer3, layer4")
        return getattr(model, ifa_layer)

    raise ValueError(f"Unsupported source model for IFA: {source_model_name}")


def make_pgd_adversary(model, loss_fn, args):
    return LinfPGDAttack(
        model,
        loss_fn=loss_fn,
        eps=args.B,
        nb_iter=args.attack_iter,
        eps_iter=args.B / 10,
        rand_init=True,
        clip_min=-2.2,
        clip_max=2.2,
        targeted=False,
    )


def make_save_path(args, attack_method):
    if attack_method == "IFA":
        attack_dir = osp.join(args.output_dir, args.source_model, args.dataset, attack_method, args.ifa_layer)
    else:
        attack_dir = osp.join(args.output_dir, args.source_model, args.dataset, attack_method)

    filename = f"eps{args.B}_iter{args.attack_iter}_m{args.m}_neuron{args.neuron_idx}.pth"
    return osp.join(attack_dir, filename)


def get_attack_methods(args):
    return [normalize_attack_method(args.attack_method)]


def get_taa_target(batch, model, args):
    with torch.no_grad():
        feature_dim = model(batch[:1]).shape[1]

    y = torch.zeros(batch.shape[0], feature_dim, device=batch.device)
    y[:, args.neuron_idx] = args.m
    return y


def generate_for_attack(args, attack_method, pretrained_model, test_loader):
    all_adv_samples, all_labels = [], []
    criterion = None

    if attack_method == "TAA":
        adversary = make_pgd_adversary(pretrained_model, taa_loss, args)
    else:
        criterion = FeatureDestructionLoss(pretrained_model, args.source_model, args.ifa_layer)
        adversary = make_pgd_adversary(pretrained_model, criterion, args)

    for i, (batch, label) in enumerate(test_loader):
        batch, label = batch.to("cuda"), label.to("cuda")

        if attack_method == "TAA":
            target = get_taa_target(batch, pretrained_model, args)
        else:
            criterion.set_clean_reference(batch)
            target = torch.zeros(batch.shape[0], device=batch.device)

        adv_batch = adversary.perturb(batch, target)
        all_adv_samples.append(adv_batch.detach().cpu())
        all_labels.append(label.detach().cpu())

        print(f"[{attack_method}] Generated batch {i + 1}/{len(test_loader)}")

    if criterion is not None:
        criterion.remove()

    save_path = make_save_path(args, attack_method)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "adv_samples": torch.cat(all_adv_samples, dim=0),
            "labels": torch.cat(all_labels, dim=0),
            "attack_method": attack_method,
            "source_model": args.source_model,
            "dataset": args.dataset,
            "ifa_layer": args.ifa_layer if attack_method == "IFA" else None,
            "eps": args.B,
            "attack_iter": args.attack_iter,
            "m": args.m,
            "neuron_idx": args.neuron_idx,
        },
        save_path,
    )

    print(f"[{attack_method}] Adversarial samples saved to: {save_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Generate TAA/IFA adversarial samples")
    parser.add_argument("--datapath", type=str, default="/data", help="path to the dataset")
    parser.add_argument("--dataset", type=str, default="CUB200Data")
    parser.add_argument("--source_model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--attack_method", type=str, default="IFA", choices=["TAA", "IFA", "taa", "ifa"])
    parser.add_argument("--ifa_layer", type=str, default="layer4", help="ResNet layer: layer1-layer4")
    parser.add_argument("--B", type=float, default=0.1, help="Attack budget (eps)")
    parser.add_argument("--attack_iter", type=int, default=40, help="Number of PGD iterations")
    parser.add_argument("--m", type=float, default=1000, help="TAA target activation value")
    parser.add_argument("--neuron_idx", type=int, default=0, help="TAA target neuron index")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="adv_samples", help="Root directory for adversarial samples")
    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_set = eval(args.dataset)(
        args.datapath,
        False,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        -1,
        98,
        preload=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    print(f"Loading source model: {args.source_model}")
    pretrained_model = eval(f"fe{args.source_model}")(pretrained=True).eval().cuda()

    for attack_method in get_attack_methods(args):
        generate_for_attack(args, attack_method, pretrained_model, test_loader)


if __name__ == "__main__":
    main()
