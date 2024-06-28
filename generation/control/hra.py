"""
This script utilizes code from lora available at: 
https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

Original Author: Simo Ryu
License: Apache License 2.0
"""


import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pickle

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


class HRAInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=8, apply_GS=False,
    ):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        
        self.r = r
        self.apply_GS = apply_GS
        
        half_u = torch.zeros(in_features, r // 2)
        nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
        self.hra_u = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1), requires_grad=True)

        self.fixed_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        orig_weight = self.fixed_linear.weight.data
        if self.apply_GS:
            weight = [(self.hra_u[:, 0] / self.hra_u[:, 0].norm()).view(-1, 1)]
            for i in range(1, self.r):
                ui = self.hra_u[:, i].view(-1, 1)
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            new_weight = orig_weight @ (torch.eye(self.in_features, device=x.device) - 2 * weight @ weight.t())
            
        else:
            new_weight = orig_weight
            hra_u_norm = self.hra_u / self.hra_u.norm(dim=0)
            for i in range(self.r):
                ui = hra_u_norm[:, i].view(-1, 1)
                new_weight = torch.mm(new_weight, torch.eye(self.in_features, device=x.device) - 2 * ui @ ui.t())

        out = nn.functional.linear(input=x, weight=new_weight, bias=self.fixed_linear.bias)
        return out


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_CONV_TARGET_REPLACE = {"ResBlock"}

UNET_EXTENDED_TARGET_REPLACE = {"ResBlock", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    result = []
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                result.append((parent, name, module))  # Append the result to the list

    return result  # Return the list instead of using 'yield'


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        HRAInjectedLinear,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # the first modules is the most senior father class.
        # this, incase you want to naively iterate over all modules.
        for module in model.modules():
            ancestor_class = module.__class__.__name__
            break
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )

    results = []
    # For each target find every linear_class module that isn't a child of a HRAInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a HRAInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                results.append((parent, name, module))  # Append the result to the list

    return results  # Return the list instead of using 'yield'

def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [HRAInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))

    return ret


_find_modules = _find_modules_v2
# _find_modules = _find_modules_old

def inject_trainable_hra(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = False,
    r: int = 8,
    apply_GS: str = False,
):
    """
    inject hra into model, and returns hra parameter groups.
    """

    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):

        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("HRA Injection : injecting hra into ", name)
            print("HRA Injection : weight shape", weight.shape)
        _tmp = HRAInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            apply_GS=apply_GS,
        )
        _tmp.fixed_linear.weight = weight
        if bias is not None:
            _tmp.fixed_linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp
        
        require_grad_params.append(_module._modules[name].hra_u)
        _module._modules[name].hra_u.requires_grad = True
        
        names.append(name)

    return require_grad_params, names
