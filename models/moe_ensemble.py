import torch
import torch.nn as nn
import models
import data.dino as dino
from utils.model import get_layers
from torchvision import models as torchvision_models


def get_straight_through_variable(x):
    assert len(x.shape) == 2
    index = x.max(1, keepdim=True)[1]
    x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
    return x_hard - x.detach() + x


class MoeEnsemble(nn.Module):
    def __init__(self, router_arch, expert_arch, expert_layer_type, expert_init_type, num_classes, num_experts=5,
                 router_patch_size=8, router_checkpoint_path=None, router_checkpoint_key="teacher",
                 routing_policy="hard"):
        super(MoeEnsemble, self).__init__()

        # define the router
        if router_arch == 'resnet50':
            self.router = torchvision_models.__dict__['resnet50']()
            if router_checkpoint_path is None:
                raise NotImplementedError
        elif router_arch == 'vit_small':
            self.router = dino.vision_transformer.__dict__[router_arch](
                patch_size=router_patch_size, num_classes=num_experts)
            if router_checkpoint_path is None:
                router_checkpoint_path = './data/dino/dino_deitsmall8_pretrain.pth'
        else:
            raise NotImplementedError
        dino.utils.load_pretrained_weights(
            self.router, router_checkpoint_path, router_checkpoint_key, router_arch, router_patch_size)

        self.routing_policy = routing_policy
        if self.routing_policy == "hard":
            self.routing_func = get_straight_through_variable
        elif self.routing_policy == "soft":
            self.routing_func = lambda t: t
        else:
            raise NotImplementedError

        # define the experts
        cl, ll = get_layers(expert_layer_type)
        self.experts = nn.ModuleList([
            models.__dict__[expert_arch](cl, ll, expert_init_type, num_classes=num_classes)
            for _ in range(num_experts)
        ])

    def forward(self, x_expert, x_router=None):
        x_router = x_expert if x_router else x_router
        a = self.routing_func(self.router(x_router))
        b = torch.cat([expert(x_expert) for expert in self.experts], dim=-1)
        return a * b
