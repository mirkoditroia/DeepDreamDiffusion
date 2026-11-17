"""Network modules. Imported only when a model is actually constructed.

TouchDesigner imports ``deepdream.backbones`` on every cook for the layer
names. Keeping torch out of that import avoids creating a CUDA context in
the same process that is drawing the frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _download_weights(url: str, filename: str, sha256: str) -> dict:
    """Fetch a checkpoint once into the torch hub cache, checked by its SHA-256.

    The file is read as tensors only (weights_only): no code in it can run.
    """
    import os

    folder = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        print(f"[DeepDream Diffusion] Downloading {filename} once from {url}", flush=True)
        torch.hub.download_url_to_file(url, path, hash_prefix=sha256, progress=False)
    return torch.load(path, map_location="cpu", weights_only=True)


class _InceptionV1(nn.Module):
    """GoogLeNet as Google trained it for the DeepDream release (inception5h).

    This is the network of the 2015 DeepDream pictures: no BatchNorm, local
    response normalization after the stem, input in 0-255 minus 117. The
    torchvision GoogLeNet, a later retrain with BatchNorm, dreamed grain and
    was removed. Weights: Google's inception5h.zip (Apache 2.0), as converted to
    PyTorch by ProGamerGov/pytorch-old-tensorflow-models.
    """

    URL = (
        "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/"
        "raw/master/inception5h.pth"
    )
    SHA256 = "44f1f1e0734fa77edd0ab948c0d4d5c9ab6f4efce10b02d793fd14ab2d71a73b"
    BLOCKS = {
        # name: (in, 1x1, 3x3 reduce, 3x3, 5x5 reduce, 5x5, pool proj)
        "mixed3a": (192, 64, 96, 128, 16, 32, 32),
        "mixed3b": (256, 128, 128, 192, 32, 96, 64),
        "mixed4a": (480, 192, 96, 204, 16, 48, 64),
        "mixed4b": (508, 160, 112, 224, 24, 64, 64),
        "mixed4c": (512, 128, 128, 256, 24, 64, 64),
        "mixed4d": (512, 112, 144, 288, 32, 64, 64),
        "mixed4e": (528, 256, 160, 320, 32, 128, 128),
        "mixed5a": (832, 256, 160, 320, 48, 128, 128),
    }
    # A stride-2 max pool comes before these blocks.
    POOLED = {"mixed4a", "mixed5a"}

    def __init__(self):
        super().__init__()
        self.layer_names = BACKBONE_LAYERS["inception_v1"]
        self.conv2d0_pre_relu_conv = nn.Conv2d(3, 64, 7, stride=2)
        self.conv2d1_pre_relu_conv = nn.Conv2d(64, 64, 1)
        self.conv2d2_pre_relu_conv = nn.Conv2d(64, 192, 3)
        for name, (cin, c1, r3, c3, r5, c5, cp) in self.BLOCKS.items():
            setattr(self, f"{name}_1x1_pre_relu_conv", nn.Conv2d(cin, c1, 1))
            setattr(self, f"{name}_3x3_bottleneck_pre_relu_conv", nn.Conv2d(cin, r3, 1))
            setattr(self, f"{name}_5x5_bottleneck_pre_relu_conv", nn.Conv2d(cin, r5, 1))
            setattr(self, f"{name}_pool_reduce_pre_relu_conv", nn.Conv2d(cin, cp, 1))
            setattr(self, f"{name}_3x3_pre_relu_conv", nn.Conv2d(r3, c3, 3))
            setattr(self, f"{name}_5x5_pre_relu_conv", nn.Conv2d(r5, c5, 5))
        state = _download_weights(self.URL, "inception5h.pth", self.SHA256)
        own = self.state_dict()
        self.load_state_dict({k: v for k, v in state.items() if k in own})
        # The engine feeds ImageNet-normalized RGB. One affine turns it into
        # the 0-255 minus 117 the network was trained on.
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.register_buffer("input_scale", std * 255.0)
        self.register_buffer("input_bias", mean * 255.0 - 117.0)
        self.stop_index = len(self.layer_names) - 1

    def set_active_layers(self, layers: list[str]) -> None:
        self.stop_index = max(self.layer_names.index(layer) for layer in layers)

    @staticmethod
    def _lrn(x: torch.Tensor) -> torch.Tensor:
        # TensorFlow LRN(depth_radius=5, bias=2, alpha=1e-4, beta=0.5), read
        # from the inception5h graph. PyTorch divides alpha by the window.
        # The stem reaches about 5500, so the squares need float32 even when
        # the rest of the network runs in half precision.
        out = F.local_response_norm(
            x.float(), size=11, alpha=11e-4, beta=0.5, k=2.0
        )
        return out.to(x.dtype)

    @staticmethod
    def _pool(x: torch.Tensor, stride: int) -> torch.Tensor:
        pad = (0, 1, 0, 1) if stride == 2 else (1, 1, 1, 1)
        return F.max_pool2d(F.pad(x, pad, value=float("-inf")), 3, stride)

    def _block(self, name: str, x: torch.Tensor) -> torch.Tensor:
        conv = lambda part: getattr(self, f"{name}_{part}_pre_relu_conv")  # noqa: E731
        a = F.relu(conv("1x1")(x))
        b = F.relu(conv("3x3")(F.pad(F.relu(conv("3x3_bottleneck")(x)), (1, 1, 1, 1))))
        c = F.relu(conv("5x5")(F.pad(F.relu(conv("5x5_bottleneck")(x)), (2, 2, 2, 2))))
        d = F.relu(conv("pool_reduce")(self._pool(x, 1)))
        return torch.cat((a, b, c, d), 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x * self.input_scale + self.input_bias
        x = F.relu(self.conv2d0_pre_relu_conv(F.pad(x, (2, 3, 2, 3))))
        x = self._lrn(self._pool(x, 2))
        x = F.relu(self.conv2d1_pre_relu_conv(x))
        x = F.relu(self.conv2d2_pre_relu_conv(F.pad(x, (1, 1, 1, 1))))
        x = self._pool(self._lrn(x), 2)
        out: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.layer_names):
            if name in self.POOLED:
                x = self._pool(x, 2)
            x = self._block(name, x)
            out[name] = x
            if index == self.stop_index:
                break
        return out


_BUILDERS = {
    "vgg16": _Vgg16,
    "vgg16_exp": _Vgg16Experimental,
    "inception_v1": _InceptionV1,
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
