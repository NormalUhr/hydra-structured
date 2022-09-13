import torch
import torch.nn as nn
import models
import data.dino as dino
from utils.model import get_layers
from torchvision import models as torchvision_models
from utils.model import set_prune_rate_model


def get_straight_through_variable(x):
    assert len(x.shape) == 2
    index = x.max(1, keepdim=True)[1]
    x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0).to(x.device)
    return x_hard - x.detach() + x


class MoeEnsemble(nn.Module):
    def __init__(self, num_classes, expert_arch, router_arch, router_checkpoint_path=None,
                 num_experts=5, expert_layer_type="subnet", expert_init_type="kaiming_normal",
                 router_patch_size=8, router_checkpoint_key="teacher", routing_policy="hard",
                 router_layer_type="subnet", router_init_type="kaiming_normal"):
        super(MoeEnsemble, self).__init__()

        # define the router
        if router_arch == 'resnet50':
            self.router = torchvision_models.__dict__['resnet50']()
            dino.utils.load_pretrained_weights(
                self.router, router_checkpoint_path, router_checkpoint_key, router_arch, router_patch_size)
        elif router_arch == 'vit_small':
            self.router = dino.vision_transformer.__dict__[router_arch](
                patch_size=router_patch_size, num_classes=num_experts)
            dino.utils.load_pretrained_weights(
                self.router, router_checkpoint_path, router_checkpoint_key, router_arch, router_patch_size)
        else:
            cl, ll = get_layers(router_layer_type)
            self.router = models.__dict__[router_arch](cl, ll, router_init_type, num_classes=num_experts)
            set_prune_rate_model(self.router, 1.0)
            checkpoint = torch.load(router_checkpoint_path, map_location="cpu")["state_dict"]
            for name, param in self.router.state_dict().items():
                if checkpoint[name].shape != param.shape:
                    checkpoint.pop(name)
            self.router.load_state_dict(checkpoint, strict=False)

        self.routing_policy = routing_policy
        if self.routing_policy == "hard":
            self.routing_func = lambda t: get_straight_through_variable(torch.softmax(t, dim=-1))
        elif self.routing_policy == "soft":
            self.routing_func = lambda t: torch.softmax(t, dim=-1)
        else:
            raise NotImplementedError

        # define the experts
        self.num_experts = num_experts
        cl, ll = get_layers(expert_layer_type)
        self.experts = nn.ModuleList([
            models.__dict__[expert_arch](cl, ll, expert_init_type, num_classes=num_classes)
            for _ in range(num_experts)
        ])

    def forward(self, x_expert, x_router=None):
        # todo: hard routing mode could be optimized.
        x_router = x_expert if x_router is None else x_router
        batch_size = x_expert.shape[0]
        a = self.routing_func(self.router(x_router)).view(batch_size, self.num_experts, 1)
        b = torch.cat([expert(x_expert).view(batch_size, 1, -1) for expert in self.experts], dim=1)
        return (a * b).sum(dim=1)
