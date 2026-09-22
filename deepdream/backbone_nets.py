"""Network modules. Imported only when a model is actually constructed.

TouchDesigner imports ``deepdream.backbones`` on every cook for the layer
names. Keeping torch out of that import avoids creating a CUDA context in
the same process that is drawing the frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .backbones import BACKBONE_LAYERS


class _Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        feats = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.layer_names = BACKBONE_LAYERS["vgg16"]
        self.slice1 = nn.Sequential(*[feats[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[feats[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[feats[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[feats[i] for i in range(16, 23)])
        self.stop_index = len(self.layer_names) - 1

    def set_active_layers(self, layers: list[str]) -> None:
        self.stop_index = max(self.layer_names.index(layer) for layer in layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        x = self.slice1(x)
        out["relu1_2"] = x
        if self.stop_index == 0:
            return out
        x = self.slice2(x)
        out["relu2_2"] = x
        if self.stop_index == 1:
            return out
        x = self.slice3(x)
        out["relu3_3"] = x
        if self.stop_index == 2:
            return out
        x = self.slice4(x)
        out["relu4_3"] = x
        return out


class _Vgg16Experimental(nn.Module):
    """VGG16 sliced so each exposed layer is one sequential block.

    The live panel asks for a single layer. The forward stops there instead of
    walking the rest of the network one Python call at a time.
    """

    def __init__(self):
        super().__init__()
        feats = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.layer_names = BACKBONE_LAYERS["vgg16_exp"]
        # Exclusive ends in vgg16.features: relu3_3, relu4_1, relu4_2,
        # relu4_3, relu5_1, relu5_2, relu5_3, mp5.
        ends = (16, 19, 21, 23, 26, 28, 30, 31)
        start = 0
        blocks = []
        for end in ends:
            blocks.append(nn.Sequential(*[feats[i] for i in range(start, end)]))
            start = end
        self.blocks = nn.ModuleList(blocks)
        self.stop_index = len(self.layer_names) - 1

    def set_active_layers(self, layers: list[str]) -> None:
        self.stop_index = max(self.layer_names.index(layer) for layer in layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        stop = self.stop_index
        for index, block in enumerate(self.blocks):
            x = block(x)
            out[self.layer_names[index]] = x
            if index == stop:
                return out
        return out


class _GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.googlenet(
            weights=models.GoogLeNet_Weights.IMAGENET1K_V1
        ).eval()
        self.layer_names = BACKBONE_LAYERS["googlenet"]
        self.conv1 = net.conv1
        self.maxpool1 = net.maxpool1
        self.conv2 = net.conv2
        self.conv3 = net.conv3
        self.maxpool2 = net.maxpool2
        self.inception3a = net.inception3a
        self.inception3b = net.inception3b
        self.maxpool3 = net.maxpool3
        self.inception4a = net.inception4a
        self.inception4b = net.inception4b
        self.inception4c = net.inception4c
        self.inception4d = net.inception4d
        self.inception4e = net.inception4e
        self.stop_index = len(self.layer_names) - 1
        # One affine instead of three slice-and-cat kernels. Same ImageNet
        # shift the stock GoogLeNet applies on top of our normalization.
        self.register_buffer(
            "input_scale",
            torch.tensor(
                [0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5], dtype=torch.float32
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_bias",
            torch.tensor(
                [
                    (0.485 - 0.5) / 0.5,
                    (0.456 - 0.5) / 0.5,
                    (0.406 - 0.5) / 0.5,
                ],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def set_active_layers(self, layers: list[str]) -> None:
        self.stop_index = max(self.layer_names.index(layer) for layer in layers)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.input_scale + self.input_bias

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._transform_input(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        # The selected layer is the last one we pay for. Later inceptions are
        # a different picture and are not a prefix of this one.
        if self.stop_index <= 0:
            return {"inception3b": x}
        inception3b = x
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        if self.stop_index <= 1:
            return {"inception3b": inception3b, "inception4c": x}
        inception4c = x
        x = self.inception4d(x)
        if self.stop_index <= 2:
            return {
                "inception3b": inception3b,
                "inception4c": inception4c,
                "inception4d": x,
            }
        inception4d = x
        x = self.inception4e(x)
        return {
            "inception3b": inception3b,
            "inception4c": inception4c,
            "inception4d": inception4d,
            "inception4e": x,
        }


_BUILDERS = {
    "vgg16": _Vgg16,
    "vgg16_exp": _Vgg16Experimental,
    "googlenet": _GoogLeNet,
}


def build_backbone(name: str, device: torch.device | str = "cuda") -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(
            f"Backbone sconosciuto: {name}. Disponibili: {sorted(_BUILDERS)}"
        )
    model = _BUILDERS[name]().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
