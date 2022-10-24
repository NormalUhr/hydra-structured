import time

import numpy as np
import torch
import torch.nn as nn

from crown.bound_layers import BoundSequential
from crown.eps_scheduler import EpsilonScheduler
from utils.eval import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.adv import trades_loss


def train(
        model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer, lr_policy
):
    num_class = 10

    sa = np.zeros((num_class, num_class - 1), dtype=np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa)
    num_steps_per_epoch = len(train_loader)
    eps_scheduler = EpsilonScheduler("linear",
                                     args.schedule_start,
                                     ((args.schedule_start + args.schedule_length) - 1) * \
                                     num_steps_per_epoch, args.starting_epsilon,
                                     args.epsilon,
                                     num_steps_per_epoch)

    end_eps = eps_scheduler.get_eps(epoch + 1, 0)
    start_eps = eps_scheduler.get_eps(epoch, 0)

    print(
        " ->->->->->->->->->-> One epoch with CROWN-IBP ({:.6f}-{:.6f})"
        " <-<-<-<-<-<-<-<-<-<-".format(start_eps, end_eps)
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ibp_losses = AverageMeter("IBP_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    ibp_acc1 = AverageMeter("IBP1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, ibp_losses, top1, ibp_acc1],
        prefix="Epoch: [{}]".format(epoch),
    )

    if not args.use_trainable_router:
        model_seq = BoundSequential.convert(model, {'same-slope': False, 'zero-lb': False, 'one-lb': False}).to(device)
        model_seq.train()
    end = time.time()

    dataloader = train_loader
    step = epoch * len(train_loader)

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        output = model_seq(images, method_opt="forward")
        ce = nn.CrossEntropyLoss()(output, target)

        eps = eps_scheduler.get_eps(epoch, i)
        # generate specifications
        c = torch.eye(num_class).type_as(images)[target].unsqueeze(1) - torch.eye(num_class).type_as(images).unsqueeze(0)
        # remove specifications to self
        I = (~(target.unsqueeze(1) == torch.arange(num_class).to(device).type_as(target).unsqueeze(0)))
        c = (c[I].view(images.size(0), num_class - 1, num_class)).to(device)
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[target].to(device)
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(images.size(0), num_class).to(device)
        ub_s = torch.zeros(images.size(0), num_class).to(device)

        data_ub = torch.min(images + eps, images.max()).to(device)
        data_lb = torch.max(images - eps, images.min()).to(device)

        ub, ilb, relu_activity, unstable, dead, alive = \
            model_seq(norm=np.inf, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")

        crown_final_beta = 0.
        beta = (args.epsilon - eps * (1.0 - crown_final_beta)) / args.epsilon

        if beta < 1e-5:
            lb = ilb
        else:
            _, _, clb, bias = model_seq(norm=np.inf, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
            # how much better is crown-ibp better than ibp?
            # diff = (clb - ilb).sum().item()
            lb = clb * beta + ilb * (1 - beta)

        lb = lb_s.scatter(1, sa_labels, lb)
        robust_ce = criterion(-lb, target)

        # print(ce, robust_ce)
        racc = accuracy(-lb, target, topk=(1,))

        loss_cert = robust_ce

        # calculate robust loss
        loss_adv = trades_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            distance=args.distance,
        )

        loss = args.mix_ratio * loss_cert + (1 - args.mix_ratio) * loss_adv

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))
        losses.update(ce.item(), images.size(0))
        ibp_losses.update(robust_ce.item(), images.size(0))
        ibp_acc1.update(racc[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_policy(step, len(train_loader))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
