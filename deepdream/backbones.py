"""Layer catalog for the DeepDream backbones.

Importing this module does not import PyTorch. TouchDesigner reads the layer
names on every cook; the networks themselves live in ``backbone_nets``.
"""

from __future__ import annotations

# Layer esposti per ogni backbone (dal piu' superficiale al piu' profondo).
BACKBONE_LAYERS: dict[str, list[str]] = {
    "vgg16": ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
    "vgg16_exp": [
        "relu3_3", "relu4_1", "relu4_2", "relu4_3",
        "relu5_1", "relu5_2", "relu5_3", "mp5",
    ],
    "googlenet": ["inception3b", "inception4c", "inception4d", "inception4e"],
}

# Layer di default consigliati per ogni backbone.
DEFAULT_LAYERS: dict[str, list[str]] = {
    "vgg16": ["relu4_3"],
    "vgg16_exp": ["relu4_3"],
    "googlenet": ["inception4c", "inception4d", "inception4e"],
}


def build_backbone(name: str, device: str = "cuda"):
    from .backbone_nets import build_backbone as _build

    return _build(name, device)
