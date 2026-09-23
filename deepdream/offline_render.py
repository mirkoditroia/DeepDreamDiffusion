"""Offline DeepDream stills and movies.

This is the slow published recipe (Inception v3, mean activation loss,
standard-deviation gradient, small steps, octave detail). It runs in its own
process. The live TouchDesigner cook does not import this module.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

_FEATURE_ORDER = (
    "Conv2d_1a_3x3",
    "Conv2d_2a_3x3",
    "Conv2d_2b_3x3",
    "maxpool1",
    "Conv2d_3b_1x1",
    "Conv2d_4a_3x3",
    "maxpool2",
    "Mixed_5b",
    "Mixed_5c",
    "Mixed_5d",
    "Mixed_6a",
    "Mixed_6b",
    "Mixed_6c",
    "Mixed_6d",
    "Mixed_6e",
    "Mixed_7a",
    "Mixed_7b",
    "Mixed_7c",
)

# Classic is the TensorFlow DeepDream tutorial pair (mixed3, mixed5).
LOOKS = {
    "classic": ("Mixed_6a", "Mixed_6c"),
    "deep": ("Mixed_6c", "Mixed_6e"),
    "fine": ("Mixed_5d", "Mixed_6a"),
}

_MIN_OCTAVE = 128
_TILE_AT = 1280
_TILE = 512


def say(message: str) -> None:
    print(message, flush=True)


def octave_sizes(
    height: int,
    width: int,
    octaves: int,
    scale: float,
    minimum: int = _MIN_OCTAVE,
) -> list[tuple[int, int]]:
    """Smallest octave first. The last size is the render size."""
    sizes: list[tuple[int, int]] = []
    for level in range(max(1, octaves)):
        factor = scale ** (level - octaves + 1)
        nh = max(minimum, int(round(height * factor)))
        nw = max(minimum, int(round(width * factor)))
        sizes.append((nh, nw))
    sizes[-1] = (max(minimum, height), max(minimum, width))
    return sizes


def fit_width(height: int, width: int, target_width: int) -> tuple[int, int]:
    if width <= 0:
        return max(1, height), max(1, target_width)
    target_height = max(1, int(round(height * float(target_width) / float(width))))
    return target_height, max(1, target_width)


class _InceptionFeatures(nn.Module):
    def __init__(self, layers: tuple[str, ...], device: torch.device):
        super().__init__()
        unknown = [name for name in layers if name not in _FEATURE_ORDER]
        if unknown:
            raise ValueError(f"Unknown Inception layers: {unknown}")
        self.layers = tuple(layers)
        self.device = device
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.net = inception_v3(weights=weights, transform_input=False)
        self.net.eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        deepest = max(_FEATURE_ORDER.index(name) for name in self.layers)
        self._run_order = _FEATURE_ORDER[: deepest + 1]
        self._wanted = set(self.layers)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        found: dict[str, torch.Tensor] = {}
        value = image
        for name in self._run_order:
            value = getattr(self.net, name)(value)
            if name in self._wanted:
                found[name] = value
        return [found[name] for name in self.layers]


def _loss_and_grad(model: _InceptionFeatures, image: torch.Tensor):
    leaf = image.detach().requires_grad_(True)
    activations = model(leaf)
    loss = torch.stack([activation.mean() for activation in activations]).sum()
    (grad,) = torch.autograd.grad(loss, leaf)
    return grad.detach(), float(loss.detach())


def _tile_starts(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _tiled_grad(model: _InceptionFeatures, image: torch.Tensor, tile: int):
    _, _, height, width = image.shape
    grad = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    loss_sum = 0.0
    count = 0
    for y in _tile_starts(height, tile):
        for x in _tile_starts(width, tile):
            piece = image[:, :, y : y + tile, x : x + tile]
            piece_grad, loss = _loss_and_grad(model, piece)
            ph, pw = piece_grad.shape[-2:]
            grad[:, :, y : y + ph, x : x + pw] += piece_grad
            weight[:, :, y : y + ph, x : x + pw] += 1
            loss_sum += loss
            count += 1
    return grad / weight.clamp(min=1), loss_sum / max(count, 1)


def _ascent(model: _InceptionFeatures, image: torch.Tensor, step_size: float):
    _, _, height, width = image.shape
    jitter = max(8, min(height, width) // 16)
    shift_y = int(torch.randint(-jitter, jitter + 1, ()).item())
    shift_x = int(torch.randint(-jitter, jitter + 1, ()).item())
    rolled = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    if max(height, width) > _TILE_AT:
        grad, loss = _tiled_grad(model, rolled, _TILE)
    else:
        grad, loss = _loss_and_grad(model, rolled)
    grad = grad / (grad.std() + 1e-8)
    updated = rolled.detach() + float(step_size) * grad
    updated = torch.roll(updated, shifts=(-shift_y, -shift_x), dims=(-2, -1))
    return updated.clamp(-1, 1), loss


def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def dream_tensor(
    model: _InceptionFeatures,
    image_01: np.ndarray,
    *,
    steps: int,
    octaves: int,
    step_size: float,
    scale: float,
    on_octave,
    on_step,
) -> np.ndarray:
    """image_01 is float RGB, top row first, in 0..1. Returns the same layout."""
    array = np.ascontiguousarray(image_01, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=model.device, dtype=torch.float32)
    tensor = tensor * 2.0 - 1.0
    _, _, height, width = tensor.shape
    sizes = octave_sizes(height, width, octaves, scale)
    original = tensor.detach()
    detail = None
    current = original
    for index, (nh, nw) in enumerate(sizes):
        with torch.no_grad():
            base = _resize(original, (nh, nw))
            if detail is None:
                current = base
            else:
                current = (base + _resize(detail, (nh, nw))).clamp(-1, 1)
        on_octave(index, len(sizes), nw, nh)
        for step in range(steps):
            current, loss = _ascent(model, current, step_size)
            on_step(index, len(sizes), step + 1, steps, loss)
        with torch.no_grad():
            detail = (current - base).detach()
    with torch.no_grad():
        out = ((current.clamp(-1, 1) + 1.0) * 0.5).squeeze(0)
        out = out.permute(1, 2, 0).detach().float().cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read the image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _resize_rgb(image: np.ndarray, width: int) -> np.ndarray:
    height, current_width = image.shape[:2]
    target_h, target_w = fit_width(height, current_width, width)
    if (target_h, target_w) == (height, current_width):
        return image
    interpolation = cv2.INTER_AREA if target_w < current_width else cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _write_rgb(path: Path, image_01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(image_01 * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not write {path}")


def _apply_mask(
    dream: np.ndarray,
    original: np.ndarray,
    mask_path: Path | None,
    channel: str,
) -> np.ndarray:
    if mask_path is None or not mask_path.is_file():
        return dream
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        say(f"Mask could not be read, rendering the full frame: {mask_path}")
        return dream
    if raw.ndim == 2:
        mask = raw.astype(np.float32) / 255.0
    elif channel == "alpha" and raw.shape[2] >= 4:
        mask = raw[:, :, 3].astype(np.float32) / 255.0
    else:
        mask = raw[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    mask = cv2.resize(mask, (dream.shape[1], dream.shape[0]), interpolation=cv2.INTER_LINEAR)
    original_sized = _resize_rgb(original, dream.shape[1])
    if original_sized.shape[0] != dream.shape[0]:
        original_sized = cv2.resize(
            original_sized,
            (dream.shape[1], dream.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    weight = np.clip(mask, 0.0, 1.0)[..., None]
    return dream * weight + original_sized * (1.0 - weight)


def _load_model(layers: tuple[str, ...], device_name: str) -> _InceptionFeatures:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        say("CUDA is not available. Rendering on CPU, which is much slower.")
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        say(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        say("Device: CPU")
    say("Loading Inception v3. The first run downloads the weights.")
    model = _InceptionFeatures(layers, device)
    say("Inception v3 is loaded.")
    return model


def render_still(args, model: _InceptionFeatures) -> None:
    source = _read_rgb(Path(args.image))
    original = source
    source = _resize_rgb(source, args.width)
    started = time.perf_counter()

    def on_octave(index, total, width, height):
        say(f"octave {index + 1}/{total}  {width}x{height}")

    def on_step(index, total, step, steps, loss):
        if step == steps or step % 10 == 0:
            say(
                f"octave {index + 1}/{total}  step {step}/{steps}  loss {loss:.3f}"
            )

    dream = dream_tensor(
        model,
        source,
        steps=args.steps,
        octaves=args.octaves,
        step_size=args.step_size,
        scale=args.scale,
        on_octave=on_octave,
        on_step=on_step,
    )
    mask = Path(args.mask) if args.mask else None
    dream = _apply_mask(dream, original, mask, args.mask_channel)
    output = Path(args.output)
    _write_rgb(output, dream)
    say(f"Saved {output}")
    say(f"Done in {time.perf_counter() - started:.1f} s")


def render_movie(args, model: _InceptionFeatures) -> None:
    source = Path(args.movie)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the movie: {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0:
        fps = float(getattr(args, "fps", 0.0) or 0.0)
    if fps <= 1.0:
        fps = 24.0
    if frame_count > 0:
        say(
            f"Source {frame_count} frames at {fps:.3f} fps "
            f"({frame_count / fps:.2f} s). The MP4 keeps that length."
        )
    else:
        say(f"Source frame rate {fps:.3f} fps. The MP4 keeps that rate.")
    output = Path(args.output)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    processed = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            processed += 1
            total_label = str(frame_count) if frame_count > 0 else "?"
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            original = rgb
            rgb = _resize_rgb(rgb, args.width)
            prefix = f"frame {processed}/{total_label}"
            say(prefix)

            def on_octave(index, total, width, height, prefix=prefix):
                say(f"{prefix}  octave {index + 1}/{total}  {width}x{height}")

            def on_step(index, total, step, steps, loss, prefix=prefix):
                if step == steps or step % 10 == 0:
                    say(
                        f"{prefix}  octave {index + 1}/{total}  "
                        f"step {step}/{steps}  loss {loss:.3f}"
                    )

            dream = dream_tensor(
                model,
                rgb,
                steps=args.steps,
                octaves=args.octaves,
                step_size=args.step_size,
                scale=args.scale,
                on_octave=on_octave,
                on_step=on_step,
            )
            mask = Path(args.mask) if args.mask else None
            dream = _apply_mask(dream, original, mask, args.mask_channel)
            bgr = cv2.cvtColor(
                np.clip(dream * 255.0, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                candidate = cv2.VideoWriter(
                    str(output),
                    fourcc,
                    fps,
                    (bgr.shape[1], bgr.shape[0]),
                )
                if not candidate.isOpened():
                    candidate.release()
                    raise RuntimeError(f"Could not open an MP4 writer for {output}")
                writer = candidate
                say(f"Writing {output} at {fps:.3f} fps")
            writer.write(bgr)
            elapsed = time.perf_counter() - started
            if frame_count > 0 and processed > 0:
                left = elapsed / processed * max(0, frame_count - processed)
                say(
                    f"{prefix} saved  {elapsed:.0f}s elapsed  "
                    f"about {left / 60.0:.1f} min left"
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if processed == 0:
        raise RuntimeError(f"The movie has no frames: {source}")
    _copy_source_audio(output, source)
    seconds = processed / fps if fps > 0 else 0.0
    say(f"Saved {output}")
    say(
        f"Done. {processed} frames at {fps:.3f} fps "
        f"({seconds:.2f} s) in {time.perf_counter() - started:.1f} s"
    )


def _copy_source_audio(video_out: Path, source: Path) -> None:
    """Keep the original soundtrack when ffmpeg can read it."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        say("The MP4 is the picture only. ffmpeg is not installed, so the original audio stays out.")
        return
    temp = video_out.with_name(video_out.stem + "_picture.mp4")
    video_out.replace(temp)
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(temp),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(video_out),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not video_out.is_file():
        if video_out.exists():
            video_out.unlink()
        temp.replace(video_out)
        say("The MP4 is the picture only. The source file has no audio track to copy.")
        return
    temp.unlink(missing_ok=True)
    say("The original audio is in the MP4.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepDream Diffusion offline render")
    parser.add_argument("--image", default="")
    parser.add_argument("--movie", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--octaves", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--look", choices=tuple(LOOKS), default="classic")
    parser.add_argument("--mask", default="")
    parser.add_argument("--mask-channel", choices=("alpha", "luminance"), default="alpha")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lock", default="")
    args = parser.parse_args(argv)
    if not args.image and not args.movie:
        parser.error("Pass --image or --movie.")
    if args.width < 64 or args.width > 4096:
        parser.error("--width must be from 64 to 4096.")
    if args.steps < 1 or args.steps > 400:
        parser.error("--steps must be from 1 to 400.")
    if args.octaves < 1 or args.octaves > 8:
        parser.error("--octaves must be from 1 to 8.")
    if not 1.05 <= args.scale <= 2.0:
        parser.error("--scale must be from 1.05 to 2.")
    if not 0.001 <= args.step_size <= 0.08:
        parser.error("--step-size must be from 0.001 to 0.08.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lock = Path(args.lock) if args.lock else None
    if lock is not None:
        lock.write_text(str(os.getpid()), encoding="ascii")
    try:
        say("DeepDream Diffusion - offline render")
        say(
            f"Look {args.look} ({', '.join(LOOKS[args.look])})  "
            f"width {args.width}  steps {args.steps}  "
            f"octaves {args.octaves}  scale {args.scale}  step {args.step_size}"
        )
        say("The live component is not used for this file.")
        model = _load_model(LOOKS[args.look], args.device)
        if args.movie:
            say(f"Movie: {args.movie}")
            render_movie(args, model)
        else:
            say(f"Image: {args.image}")
            render_still(args, model)
    finally:
        if lock is not None:
            try:
                lock.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
