"""Engine DeepDream fedele a gordicaleksa/pytorch-deepdream.

Differenze chiave rispetto al tutorial TensorFlow (e dal nostro engine Inception):
1. Loss = MSE delle attivazioni (media dei quadrati) -> amplificazione piu' forte.
2. Smoothing gaussiano a cascata dei gradienti -> pattern lisci e "onirici".
3. Jitter (shift circolare casuale) prima di ogni step -> niente artefatti.
4. Image pyramid (pyramid_size / pyramid_ratio) invece delle ottave classiche.
5. Backbone selezionabili: VGG16 (astratto) o GoogLeNet (cani/occhi).
"""

from __future__ import annotations

import math
import numbers
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DEFAULT_LAYERS, build_backbone
from .borders import empty_border_mask, restore_empty_border

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CascadeGaussianSmoothing(nn.Module):
    """Smoothing gaussiano depthwise a 3 sigma (0.5x, 1x, 2x), come gordicaleksa."""

    def __init__(self, kernel_size: int, sigma: float, device: torch.device | str):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]
        sigmas = [[c * sigma, c * sigma] for c in cascade_coefficients]
        self.pad = int(kernel_size[0] / 2)

        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )

        kernels = []
        for sigma_pair in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma_pair, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= (
                    1 / (std_1d * math.sqrt(2 * math.pi))
                    * torch.exp(-(((grid - mean) / std_1d) ** 2) / 2)
                )
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            kernel = kernel / torch.sum(kernel)
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1).to(device)
            gaussian_kernels.append(kernel)

        # Convolution is linear, so averaging the three normalized kernels
        # first is equivalent to averaging three convolution outputs.
        combined = torch.stack(gaussian_kernels).mean(dim=0)
        self.register_buffer("weight", combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [self.pad] * 4, mode="reflect")
        c = x.shape[1]
        return F.conv2d(x, weight=self.weight, groups=c)


def _hold_void(
    tensor: torch.Tensor, source: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """support 1 = picture libera, 0 = pixel noto tenuto uguale alla sorgente."""
    return tensor * support + source * (1.0 - support)


def _jitter_shift(level: int, step: int, radius: int) -> tuple[int, int]:
    """Same shift for the same step on every frame.

    A fresh random roll each frame moves the dream even when the picture
    does not, which is the flicker a temporal blend cannot fully hide.
    """
    if radius <= 0:
        return 0, 0
    n = (level + 1) * 131 + (step + 1)
    span = radius * 2 + 1
    h = (n * 1103515245 + 12345) & 0x7FFFFFFF
    w = (h * 1103515245 + 12345) & 0x7FFFFFFF
    return (h % span) - radius, (w % span) - radius


def _random_circular_shift(
    tensor: torch.Tensor, h_shift: int, w_shift: int, undo: bool = False
) -> torch.Tensor:
    if undo:
        h_shift, w_shift = -h_shift, -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
    rolled.requires_grad_(True)
    return rolled


class _AscentGraph:
    """One CUDA graph for a single gradient-ascent step at a fixed resolution.

    TouchDesigner owns a CUDA context beside PyTorch. Every small kernel pays
    for a context switch, so an eager VGG step that is ~15 ms on its own falls
    to a handful of frames per second inside the process. Replaying the whole
    ascent as one graph is a single launch.
    """

    def __init__(self, dreamer: GordicDream, height: int, width: int):
        self.dreamer = dreamer
        self.height = height
        self.width = width
        device = dreamer.device
        image = torch.zeros(
            (1, 3, height, width), device=device, dtype=torch.float32
        )
        self.static_x = image.contiguous(memory_format=torch.channels_last)
        self.static_x.requires_grad_(True)
        self.static_out = torch.empty_like(self.static_x)
        self.static_scale = torch.ones((), device=device, dtype=torch.float32)
        self.static_support = torch.ones(
            (1, 1, height, width), device=device, dtype=torch.float32
        )
        self.smoother = CascadeGaussianSmoothing(9, 1.0, device)
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _step(self) -> None:
        # autograd.grad returns a fresh tensor instead of AccumulateGrad into
        # .grad. The leaf's AccumulateGrad node is what invalidates capture
        # when warmup and capture use different streams.
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, cache_enabled=False
        ):
            loss = self.dreamer._loss(self.static_x, self.static_support)
        (gradient,) = torch.autograd.grad(loss, self.static_x)
        with torch.no_grad():
            smooth = self.smoother(gradient)
            scaled = self.dreamer._scale_gradient(smooth, self.static_support)
            updated = torch.clamp(
                self.static_x.detach() + self.static_scale * scaled,
                min=self.dreamer.lower_bound,
                max=self.dreamer.upper_bound,
            )
            self.static_out.copy_(updated)

    def _capture(self) -> None:
        if hasattr(self.dreamer.model, "set_active_layers"):
            self.dreamer.model.set_active_layers(self.dreamer.layers)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self._step()

    def replay(
        self,
        image: torch.Tensor,
        scale: float,
        sigma: float,
        support: torch.Tensor | None,
    ) -> torch.Tensor:
        kernel = self.dreamer._get_smoothing(sigma)
        with torch.no_grad():
            self.smoother.weight.copy_(kernel.weight)
            self.static_scale.fill_(scale)
            self.static_x.copy_(image)
            if support is None:
                self.static_support.fill_(1.0)
            else:
                self.static_support.copy_(support)
            self.graph.replay()
            return self.static_out


class GordicDream:
    """DeepDream con backbone selezionabile e algoritmo gordicaleksa."""

    def __init__(
        self,
        model_name: str = "vgg16",
        layers: list[str] | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.model = build_backbone(model_name, self.device)
        self.use_amp = self.device.type == "cuda"
        if self.device.type == "cuda":
            # Pyramid and live-resolution changes produce many input shapes.
            # cuDNN benchmarking blocks while testing algorithms for every new
            # shape, which looks like a TouchDesigner freeze.
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model.to(memory_format=torch.channels_last)
        self.model_name = model_name
        self.layers = layers or DEFAULT_LAYERS[model_name]

        unknown = [l for l in self.layers if l not in self.model.layer_names]
        if unknown:
            raise ValueError(
                f"Layer non validi per {model_name}: {unknown}. "
                f"Disponibili: {self.model.layer_names}"
            )
        if hasattr(self.model, "set_active_layers"):
            self.model.set_active_layers(self.layers)

        self.mean = torch.as_tensor(
            IMAGENET_MEAN, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.as_tensor(
            IMAGENET_STD, device=self.device
        ).view(1, 3, 1, 1)
        self.lower_bound = -self.mean / self.std
        self.upper_bound = (1 - self.mean) / self.std

        # Cache dei kernel di smoothing: sigma dipende solo da (iter, num_iter,
        # smoothing_coeff), quindi sono identici a ogni frame -> grosso risparmio.
        self._smoothing_cache: dict[float, CascadeGaussianSmoothing] = {}
        self._ascent_graphs: dict[tuple, _AscentGraph] = {}
        self._graph_failed: set[tuple] = set()

    def release_cuda_graphs(self) -> None:
        self._ascent_graphs.clear()

    def _get_ascent_graph(self, height: int, width: int) -> _AscentGraph | None:
        if self.device.type != "cuda":
            return None
        key = (tuple(self.layers), height, width)
        cached = self._ascent_graphs.get(key)
        if cached is not None:
            return cached
        if key in self._graph_failed:
            return None
        try:
            graph = _AscentGraph(self, height, width)
        except Exception as exc:
            self._graph_failed.add(key)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            print(
                "[DeepDreamLive] CUDA graph unavailable at "
                f"{width}x{height}, eager step will be used: {exc}"
            )
            return None
        if len(self._ascent_graphs) >= 6:
            self._ascent_graphs.pop(next(iter(self._ascent_graphs)))
        self._ascent_graphs[key] = graph
        return graph

    def _get_smoothing(self, sigma: float) -> CascadeGaussianSmoothing:
        key = round(sigma, 4)
        cached = self._smoothing_cache.get(key)
        if cached is None:
            cached = CascadeGaussianSmoothing(9, sigma, self.device)
            self._smoothing_cache[key] = cached
        return cached

    def _loss(
        self, input_tensor: torch.Tensor, support: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hasattr(self.model, "set_active_layers"):
            self.model.set_active_layers(self.layers)
        out = self.model(input_tensor)
        activations = [out[name] for name in self.layers]
        if support is None:
            losses = [act.square().mean() for act in activations]
            return torch.mean(torch.stack(losses))

        # La loss guarda solo la picture. Il nero di bordo non e' un obiettivo:
        # il gradiente non viene pagato per inventarci delle feature.
        losses = []
        for act in activations:
            weight = F.interpolate(
                support,
                size=act.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            denom = (weight.sum() * act.shape[1]).clamp_min(1.0)
            losses.append((act.square() * weight).sum() / denom)
        return torch.mean(torch.stack(losses))

    def _scale_gradient(
        self, grad: torch.Tensor, support: torch.Tensor | None
    ) -> torch.Tensor:
        if support is None:
            g_std, g_mean = torch.std_mean(grad)
            return (grad - g_mean) / (g_std + 1e-8)

        # Media e deviazione solo sui pixel della picture. Sottrarre la media
        # sull'intero frame assegnava uno step pieno anche dove il gradiente
        # era zero: e' questo che dipinge le bande nere come un filtro.
        weight = support
        count = (weight.sum() * grad.shape[1]).clamp_min(1.0)
        g_mean = (grad * weight).sum() / count
        centered = (grad - g_mean) * weight
        variance = centered.square().sum() / (count - 1.0).clamp_min(1.0)
        return centered / (variance.sqrt() + 1e-8)

    def _gradient_ascent_step(
        self,
        input_tensor: torch.Tensor,
        iteration: int,
        num_iterations: int,
        lr: float,
        smoothing_coefficient: float,
        intensity: float = 1.0,
        support: torch.Tensor | None = None,
    ) -> None:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            loss = self._loss(input_tensor, support)
        loss.backward()
        grad = input_tensor.grad.detach()

        sigma = ((iteration + 1) / num_iterations) * 2.0 + smoothing_coefficient
        smooth_grad = self._get_smoothing(sigma)(grad)
        smooth_grad = self._scale_gradient(smooth_grad, support)

        with torch.no_grad():
            input_tensor.add_(lr * intensity * smooth_grad)
            input_tensor.clamp_(min=self.lower_bound, max=self.upper_bound)
        input_tensor.grad = None

    def dream_tensor(
        self,
        img01: np.ndarray,
        *,
        pyramid_size: int = 4,
        pyramid_ratio: float = 1.8,
        num_iterations: int = 10,
        lr: float = 0.09,
        intensity: float = 1.0,
        spatial_shift_size: int = 32,
        smoothing_coefficient: float = 0.5,
        progress: Callable[[int, int], None] | None = None,
        content_source: np.ndarray | None = None,
        use_cuda_graph: bool = False,
    ) -> np.ndarray:
        """Apply DeepDream to an RGB NumPy image in [0,1], returning the same format.

        `content_source` is the picture before feedback. Flat black connected to
        the frame border is a known region: it stays equal to that source and is
        excluded from the loss, the same way a diffusion sampler holds known pixels.
        """
        base_h, base_w = img01.shape[:2]
        img = np.ascontiguousarray(img01, dtype=np.float32)
        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        if self.device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        tensor = (tensor - self.mean) / self.std

        guide01 = img01 if content_source is None else content_source
        if guide01.shape[:2] != img01.shape[:2]:
            guide01 = img01
        void = empty_border_mask(guide01)
        support_base = None
        guide_norm = None
        if void is not None:
            support_np = np.ascontiguousarray(
                (~void).astype(np.float32)
            )
            support_base = (
                torch.from_numpy(support_np)[None, None].to(self.device)
            )
            guide = np.ascontiguousarray(guide01[..., :3], dtype=np.float32)
            guide_tensor = (
                torch.from_numpy(guide)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            guide_norm = (guide_tensor - self.mean) / self.std

        for level in range(pyramid_size):
            exponent = level - pyramid_size + 1
            scale = pyramid_ratio ** exponent
            new_h = max(1, int(round(base_h * scale)))
            new_w = max(1, int(round(base_w * scale)))
            with torch.no_grad():
                tensor = F.interpolate(
                    tensor.detach(),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                support_level = None
                source_level = None
                if support_base is not None and guide_norm is not None:
                    support_level = F.interpolate(
                        support_base, size=(new_h, new_w), mode="nearest"
                    )
                    source_level = F.interpolate(
                        guide_norm,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    tensor = _hold_void(tensor, source_level, support_level)
                if self.device.type == "cuda":
                    tensor = tensor.contiguous(
                        memory_format=torch.channels_last
                    )
            tensor.requires_grad_(True)
            graph = (
                self._get_ascent_graph(new_h, new_w)
                if use_cuda_graph
                else None
            )

            for it in range(num_iterations):
                h_shift, w_shift = _jitter_shift(
                    level, it, spatial_shift_size
                )
                sigma = (
                    (it + 1) / num_iterations
                ) * 2.0 + smoothing_coefficient
                if graph is not None:
                    with torch.no_grad():
                        rolled = torch.roll(
                            tensor, shifts=(h_shift, w_shift), dims=(2, 3)
                        )
                        rolled_support = support_level
                        if support_level is not None:
                            rolled_support = torch.roll(
                                support_level,
                                shifts=(h_shift, w_shift),
                                dims=(2, 3),
                            )
                        updated = graph.replay(
                            rolled,
                            lr * intensity,
                            sigma,
                            rolled_support,
                        )
                        tensor = torch.roll(
                            updated,
                            shifts=(-h_shift, -w_shift),
                            dims=(2, 3),
                        )
                        if (
                            support_level is not None
                            and source_level is not None
                        ):
                            tensor = _hold_void(
                                tensor, source_level, support_level
                            )
                else:
                    tensor = _random_circular_shift(tensor, h_shift, w_shift)
                    rolled_support = support_level
                    if support_level is not None:
                        rolled_support = torch.roll(
                            support_level,
                            shifts=(h_shift, w_shift),
                            dims=(2, 3),
                        )
                    self._gradient_ascent_step(
                        tensor,
                        it,
                        num_iterations,
                        lr,
                        smoothing_coefficient,
                        intensity,
                        rolled_support,
                    )
                    tensor = _random_circular_shift(
                        tensor, h_shift, w_shift, undo=True
                    )
                    if support_level is not None and source_level is not None:
                        with torch.no_grad():
                            held = _hold_void(
                                tensor.detach(), source_level, support_level
                            )
                            if self.device.type == "cuda":
                                held = held.contiguous(
                                    memory_format=torch.channels_last
                                )
                        tensor = held.requires_grad_(True)
                if progress is not None:
                    progress(level, it)

        with torch.no_grad():
            output = (
                (tensor * self.std + self.mean)
                .clamp_(0.0, 1.0)
                .squeeze(0)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        if void is not None:
            source_rgb = np.ascontiguousarray(guide01[..., :3], dtype=np.float32)
            output = np.where(void[..., None], source_rgb, output)
        return output
