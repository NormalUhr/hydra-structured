from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

import torch

import models
from utils.model import (
    get_layers,
    set_prune_rate_model,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")

    parser.add_argument(
        "--result-dir",
        default="./results",
        type=str,
        help="directory to save results",
    )

    parser.add_argument(
        "--result-file",
        default="debug.pth.tar",
        type=str,
        help="output file name"
    )

    parser.add_argument("--arch",
                        default="resnet20s",
                        type=str,
                        help="Model architecture")

    parser.add_argument(
        "--layer-type",
        type=str,
        default="subnet",
        choices=("dense", "subnet"),
        help="dense | subnet"
    )

    parser.add_argument(
        "--k",
        type=float,
        default=0.5,
        help="Fraction of weight variables kept in subnet",
    )

    parser.add_argument(
        "--source-net",
        type=str,
        default="",
        help="Checkpoint which will be pruned/fine-tuned",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in the model",
    )

    args = parser.parse_args()

    cl, ll = get_layers(args.layer_type)
    model = models.__dict__[args.arch](
        cl, ll, "kaiming_normal", num_classes=args.num_classes
    )

    set_prune_rate_model(model, args.k)

    if os.path.isfile(args.source_net):
        print("=> loading source model from '{}'".format(args.source_net))
        checkpoint = torch.load(args.source_net, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.source_net))
    else:
        raise ValueError

    mask_list = []

    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            scores = v.popup_scores.abs()

            score_L1_norm = torch.norm(scores.flatten(start_dim=1, end_dim=-1), p=1, dim=1)
            _, idx = score_L1_norm.sort()
            j = int((1 - args.k) * scores.shape[0])

            # flat_out and out access the same memory.
            out = scores.clone()
            flat_out = out.flatten(start_dim=1, end_dim=-1)  # share the same memory
            flat_out[idx[:j], :] = 0
            flat_out[idx[j:], :] = 1
            flat_out = (torch.sum(flat_out.flatten(start_dim=1, end_dim=-1), dim=1) > 0).int().tolist()
            mask_list.append(flat_out)

    result_path = os.path.join(args.result_dir, args.result_file)

    print(f"Saving model mask in {args.result_file}")

    torch.save(mask_list, result_path)
