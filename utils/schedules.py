import torch
import torch.nn as nn

import numpy as np


def get_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file. 
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd,)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
    return optim


def new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_schedule(optimizer, args):
    def set_lr(step, step_per_epoch, lr=args.lr):
        warmup_step = args.warmup_epochs * step_per_epoch
        total_cosine_step = (args.epochs - args.warmup_epochs) * step_per_epoch
        if step < warmup_step:
            a = lr * step / warmup_step
        else:
            cos_step = step - warmup_step
            a = lr * 0.5 * (1 + np.cos(cos_step / total_cosine_step * np.pi))

        new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, args):
    def set_lr(step, step_per_epoch, lr=args.lr):
        warmup_step = args.warmup_epochs * step_per_epoch
        total_step = (args.epochs - args.warmup_epochs) * step_per_epoch

        if step < warmup_step:
            a = lr * step / warmup_step
        else:
            a = lr
            if step >= 0.5 * total_step:
                a *= 0.1
            if step >= 0.75 * total_step:
                a *= 0.01

        new_lr(optimizer, a)

    return set_lr
