from __future__ import absolute_import
from __future__ import print_function

import importlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import data
import models
from args import parse_args
from models.moe_ensemble import MoeEnsemble
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    current_model_pruned_fraction,
    snip_init,
)
from utils.schedules import get_lr_policy, get_optimizer
from utils.semisup import get_semisup_dataloader

# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning

global step


def main():
    args = parse_args()
    if args.configs is not None and os.path.isfile(args.configs):
        parse_configs_file(args)

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.arch, args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_trainer-{}_epochs-{}".format(
                n,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "0--k-{:.2f}_trainer-{}_epochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create model
    if not args.use_trainable_router:
        cl, ll = get_layers(args.layer_type)
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
        ).to(device)
    else:
        model = MoeEnsemble(
            router_arch=args.router_arch,
            expert_arch=args.arch,
            expert_layer_type=args.layer_type,
            expert_init_type=args.init_type,
            num_classes=args.num_classes,
            router_checkpoint_path=args.router_checkpoint_path
        ).to(device)
    # logger.info(model)

    # Customize models for training/pruning/fine-tuning
    if not args.use_trainable_router:
        prepare_model(model, args)
    else:
        for m in model.experts:
            prepare_model(m, args)

    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    # logger.info(args.dataset, D, len(train_loader.dataset), len(test_loader.dataset))

    # Semi-sup dataloader
    if args.is_semisup:
        logger.info("Using semi-supervised training")
        sm_loader = get_semisup_dataloader(args, D.tr_train)
    else:
        sm_loader = None

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    if args.val_method == "ibp":
        adv_val = getattr(importlib.import_module("utils.eval"), "ibp")
    else:
        adv_val = getattr(importlib.import_module("utils.eval"), "adv")

    std_val = getattr(importlib.import_module("utils.eval"), "base")

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            if args.use_trainable_router and args.exp_mode == "prune":
                for expert in model.experts:
                    expert.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.source_net))
            raise ValueError

    if args.exp_mode == "prune":

        # Init scores once source net is loaded.
        # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
        if args.scaled_score_init:
            if not args.use_trainable_router:
                initialize_scaled_score(model)
            else:
                for m in model.experts:
                    initialize_scaled_score(m)

        # Scaled random initialization. Useful when training a high sparse net from scratch.
        # If not used, a sparse net (without batch-norm) from scratch will not coverge.
        # With batch-norm its not really necessary.
        elif args.scale_rand_init:
            if not args.use_trainable_router:
                scale_rand_init(model, args.k)
            else:
                for m in model.experts:
                    scale_rand_init(m, args.k)

        if args.snip_init:
            if not args.use_trainable_router:
                snip_init(model, criterion, optimizer, train_loader, device, args)
            else:
                raise NotImplementedError

    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )

    best_prec1 = 0

    # resume (if checkpoint provided). Continue training with previous settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError

    # Evaluate
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        std_acc, _ = std_val(model, device, test_loader, criterion, args)
        adv_acc, _ = adv_val(model, device, test_loader, criterion, args)
        logger.info(
                f"Evaluation only: SA: {std_acc: .2f}%, RA: {adv_acc: .2f}%"
            )
        if args.evaluate:
            return

    # Start training
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):

        start = time.time()

        # train
        trainer(
            model,
            device,
            train_loader,
            sm_loader,
            criterion,
            optimizer,
            epoch,
            args,
            None,
            lr_policy,
        )

        # evaluate on test set
        std_acc, _ = std_val(model, device, test_loader, criterion, args)
        adv_acc, _ = adv_val(model, device, test_loader, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = adv_acc > best_prec1
        best_prec1 = max(adv_acc, best_prec1)
        print(
            f"Epoch {epoch}, SA: {std_acc: .2f}%, RA: {adv_acc: .2f}%, best performance (RA): {best_prec1: .2f}"
        )
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

    current_model_pruned_fraction(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
