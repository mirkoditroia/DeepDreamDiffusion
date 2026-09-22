"""Motore live DeepDream condiviso (OpenCV standalone + TouchDesigner)."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .backbones import BACKBONE_LAYERS, DEFAULT_LAYERS


def enhance_saturation(img01: np.ndarray, saturation: float) -> np.ndarray:
    if abs(saturation - 1.0) < 1e-3:
        return img01
    gray = img01.mean(axis=2, keepdims=True)
    return np.clip(gray + (img01 - gray) * saturation, 0.0, 1.0)


def apply_feedback_transform(
    img01: np.ndarray, zoom: float, rotate_deg: float
) -> np.ndarray:
    if zoom == 0.0 and rotate_deg == 0.0:
        return img01
    h, w = img01.shape[:2]
    scale = max(0.05, 1.0 + zoom)
    center = ((w - 1) * 0.5, (h - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, rotate_deg, scale)
    return cv2.warpAffine(
        img01,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _flip_rows(arr: np.ndarray) -> np.ndarray:
    """TouchDesigner stores row 0 at the bottom. ImageNet models use row 0 at the top."""
    return np.ascontiguousarray(arr[::-1])


def top_to_rgb01(arr: np.ndarray) -> np.ndarray:
    """Converte numpy da TOP TD (H,W,3|4 float 0-1 o uint8) in RGB [0,1] dall'alto."""
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32) / 255.0
    rgb = arr[..., :3].astype(np.float32, copy=False)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return _flip_rows(np.clip(rgb, 0.0, 1.0))


def top_mask_to_01(arr: np.ndarray, source: str = "alpha") -> np.ndarray:
    """Estrae una maschera [H,W,1] dal canale alpha o dalla luminanza RGB."""
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32, copy=False)
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 2:
        mask = arr
    elif source == "alpha" and arr.shape[2] >= 4:
        mask = arr[..., 3]
    elif arr.shape[2] == 1:
        mask = arr[..., 0]
    else:
        mask = (
            arr[..., 0] * 0.2126
            + arr[..., 1] * 0.7152
            + arr[..., 2] * 0.0722
        )
    return _flip_rows(np.clip(mask, 0.0, 1.0)[..., None])


def composite_with_mask(
    original01: np.ndarray,
    dreamed01: np.ndarray,
    mask01: np.ndarray | None,
) -> np.ndarray:
    """Bianco/alpha 1 = dream; nero/alpha 0 = video originale."""
    if mask01 is None:
        return dreamed01
    return original01 * (1.0 - mask01) + dreamed01 * mask01


def composite_effect_delta(
    current01: np.ndarray,
    dreamed01: np.ndarray,
    dream_source01: np.ndarray | None,
    mask01: np.ndarray | None,
) -> np.ndarray:
    """Apply only the aligned dream residual through a current-frame mask.

    A complete asynchronous dream contains an older source frame. Subtracting
    that exact source prevents moving subjects from leaving bright silhouettes
    when the result is combined with a newer video frame.
    """
    if mask01 is None:
        return dreamed01
    if (
        dream_source01 is None
        or dream_source01.shape != dreamed01.shape
        or current01.shape != dreamed01.shape
    ):
        return composite_with_mask(current01, dreamed01, mask01)
    effect_delta = dreamed01 - dream_source01
    return np.clip(current01 + effect_delta * mask01, 0.0, 1.0)


def rgb01_to_top_rgba(img01: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """RGB [0,1] dall'alto -> array TOP (H,W,4) float32, riga 0 in basso."""
    h, w = img01.shape[:2]
    out = np.ones((h, w, 4), dtype=np.float32)
    out[..., :3] = np.clip(img01, 0.0, 1.0)
    out[..., 3] = alpha
    return _flip_rows(out)


_FLOW_WIDTH = 256


def _motion_amount(current: np.ndarray, previous: np.ndarray) -> float:
    """Mean absolute change on a tiny copy. A full-frame mean is slower than the flow."""
    current_small = cv2.resize(current, (64, 36), interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous, (64, 36), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(current_small - previous_small)))


def _gray_u8(img01: np.ndarray) -> np.ndarray:
    rgb = img01[..., :3]
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)


class TemporalTracker:
    """Motion-compensated mix of the new dream with the previous one.

    Optical flow runs on a small grayscale copy. Where the warp does not
    match the picture, the new dream is kept, so a cut does not leave a ghost.
    """

    def __init__(self) -> None:
        self._dis = None
        self._guess: np.ndarray | None = None
        self._grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._guess = None

    def blend(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        current_src: np.ndarray,
        previous_src: np.ndarray,
        amount: float,
    ) -> np.ndarray:
        if amount <= 0.0 or current.shape != previous.shape:
            return current
        if _motion_amount(current_src, previous_src) < 0.012:
            self._guess = None
            if amount >= 1.0:
                return previous
            return cv2.addWeighted(current, 1.0 - amount, previous, float(amount), 0.0)
        warped, confidence = self._align(previous, current_src, previous_src)
        weight = np.clip(amount * confidence, 0.0, 1.0)[..., None]
        mixed = current * (1.0 - weight) + warped * weight
        return np.clip(mixed, 0.0, 1.0).astype(np.float32, copy=False)

    def _align(
        self,
        previous: np.ndarray,
        current_src: np.ndarray,
        previous_src: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = previous.shape[:2]
        scale = max(width, height) / _FLOW_WIDTH
        if scale < 1.0:
            scale = 1.0
        small_w = max(8, int(round(width / scale)))
        small_h = max(8, int(round(height / scale)))
        prev_small = cv2.resize(
            previous_src, (small_w, small_h), interpolation=cv2.INTER_AREA
        )
        curr_small = cv2.resize(
            current_src, (small_w, small_h), interpolation=cv2.INTER_AREA
        )
        # Backward flow: for each current pixel, where it came from in the previous frame.
        flow = self._flow(_gray_u8(curr_small), _gray_u8(prev_small))
        if flow is None:
            level = max(0.0, 1.0 - _motion_amount(current_src, previous_src) / 0.2)
            return previous, np.full((height, width), level, dtype=np.float32)
        warped_src = self._remap(prev_small, flow)
        err = np.mean(np.abs(warped_src - curr_small), axis=2)
        confidence = np.clip(1.0 - err / 0.18, 0.0, 1.0).astype(np.float32)
        confidence = cv2.GaussianBlur(confidence, (0, 0), 1.2)
        if (small_h, small_w) != (height, width):
            confidence = cv2.resize(
                confidence, (width, height), interpolation=cv2.INTER_LINEAR
            )
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= width / small_w
            flow[..., 1] *= height / small_h
        return self._remap(previous, flow), confidence

    def _flow(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
        if self._dis is None:
            create = getattr(cv2, "DISOpticalFlow_create", None)
            if create is None:
                return None
            preset = getattr(cv2, "DISOPTICAL_FLOW_PRESET_ULTRAFAST", 0)
            self._dis = create(preset)
            if hasattr(self._dis, "setFinestScale"):
                self._dis.setFinestScale(2)
            if hasattr(self._dis, "setUseSpatialPropagation"):
                self._dis.setUseSpatialPropagation(True)
        guess = self._guess
        if guess is not None and guess.shape[:2] != previous.shape[:2]:
            guess = None
        flow = self._dis.calc(previous, current, guess)
        self._guess = flow
        return flow

    def _remap(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        cached = self._grids.get((height, width))
        if cached is None:
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            cached = (grid_x, grid_y)
            self._grids[(height, width)] = cached
        grid_x, grid_y = cached
        return cv2.remap(
            img,
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def resize_rgb01(img01: np.ndarray, proc_width: int) -> np.ndarray:
    h, w = img01.shape[:2]
    if w <= proc_width:
        return img01
    proc_h = int(round(h * (proc_width / w)))
    return cv2.resize(img01, (proc_width, proc_h), interpolation=cv2.INTER_AREA)


@dataclass
class DreamControls:
    lr: float = 0.09
    intensity: float = 1.0
    iterations: int = 6
    pyramid: int = 3
    blend: float = 0.35
    feedback: float = 0.0
    zoom: float = 0.0
    rotate: float = 0.0
    saturation: float = 1.2
    layer: str | None = None


class LiveDreamEngine:
    """Processa frame RGB [0,1] con GordicDream e stato temporale (feedback/blend)."""

    def __init__(
        self,
        model_name: str = "vgg16",
        device: str | None = None,
        layers: list[str] | None = None,
    ):
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_cuda_graph = False
        self.model_name = model_name
        self.layers = layers or DEFAULT_LAYERS[model_name]
        self._dreamers: dict[str, GordicDream] = {}
        self._load_futures: dict[str, Future[GordicDream]] = {}
        self._load_errors: dict[str, str] = {}
        self._loader = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deepdream-model-loader"
        )
        self.status = "Ready"
        self._failed_config: tuple | None = None
        self.last_dream01: np.ndarray | None = None
        self.last_clean01: np.ndarray | None = None
        self._temporal = TemporalTracker()

    def set_model(self, model_name: str, layer: str | None = None) -> None:
        if model_name not in BACKBONE_LAYERS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        if layer and layer in BACKBONE_LAYERS[model_name]:
            self.layers = [layer]
        else:
            self.layers = DEFAULT_LAYERS[model_name]
        self._load_errors.pop(model_name, None)
        self._failed_config = None
        self.reset_feedback()

    def close(self) -> None:
        """Release cached models without waiting for a pending background load."""
        self._loader.shutdown(wait=False, cancel_futures=True)
        self._load_futures.clear()
        self._dreamers.clear()
        self.reset_feedback()
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_feedback(self) -> None:
        self.last_dream01 = None
        self.last_clean01 = None
        self._temporal.reset()

    def load_model_now(self, layer: str) -> str | None:
        """Load the active model on this thread. The GPU worker uses this so
        CUDA context and graph capture stay on the thread that dreams.
        """
        from .gordic import GordicDream

        try:
            dreamer = GordicDream(
                model_name=self.model_name,
                layers=[layer],
                device=self.device,
            )
        except Exception as exc:
            message = f"Could not load {self.model_name}: {exc}"
            self._load_errors[self.model_name] = message
            self.status = message
            return message
        self._load_futures.pop(self.model_name, None)
        self._load_errors.pop(self.model_name, None)
        self._dreamers[self.model_name] = dreamer
        dreamer.layers = [layer]
        self.status = f"{self.model_name} ready"
        return None

    def _get_dreamer(self, layer: str) -> GordicDream | None:
        from .gordic import GordicDream

        dreamer = self._dreamers.get(self.model_name)
        if dreamer is not None:
            dreamer.layers = [layer]
            return dreamer

        error = self._load_errors.get(self.model_name)
        if error is not None:
            self.status = error
            return None

        future = self._load_futures.get(self.model_name)
        if future is None:
            model_name = self.model_name
            future = self._loader.submit(
                GordicDream,
                model_name=model_name,
                layers=[layer],
                device=self.device,
            )
            self._load_futures[model_name] = future
            self.status = f"Loading {model_name} in background..."
            return None

        if not future.done():
            self.status = f"Loading {self.model_name} in background..."
            return None

        try:
            dreamer = future.result()
        except Exception as exc:
            message = f"Could not load {self.model_name}: {exc}"
            self._load_errors[self.model_name] = message
            self._load_futures.pop(self.model_name, None)
            self.status = message
            return None

        self._load_futures.pop(self.model_name, None)
        self._dreamers[self.model_name] = dreamer
        dreamer.layers = [layer]
        self.status = f"{self.model_name} ready"
        return dreamer

    def process_frame(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int = 416,
    ) -> np.ndarray:
        """img01 RGB float [0,1] -> dreamed RGB float [0,1] (stessa risoluzione proc)."""
        img01 = resize_rgb01(img01, proc_width)
        clean01 = img01
        layer = controls.layer or DEFAULT_LAYERS[self.model_name][0]
        dreamer = self._get_dreamer(layer)
        if dreamer is None:
            return img01

        config = (
            self.model_name,
            layer,
            img01.shape[:2],
            max(1, controls.pyramid),
            max(1, controls.iterations),
        )
        if self._failed_config == config:
            return img01

        previous_dream = None
        if self.last_dream01 is not None:
            previous_dream = apply_feedback_transform(
                self.last_dream01, controls.zoom, controls.rotate
            )
        if controls.feedback > 0.0 and previous_dream is not None:
            if previous_dream.shape == img01.shape:
                img01 = (
                    (1.0 - controls.feedback) * img01
                    + controls.feedback * previous_dream
                )

        try:
            dreamed01 = dreamer.dream_tensor(
                img01,
                pyramid_size=max(1, controls.pyramid),
                pyramid_ratio=1.6,
                num_iterations=max(1, controls.iterations),
                lr=controls.lr,
                intensity=controls.intensity,
                spatial_shift_size=16,
                smoothing_coefficient=0.5,
                content_source=clean01,
                # CUDA graph capture in TouchDesigner's process takes the
                # whole application down. The isolated GPU process opts in.
                use_cuda_graph=self.use_cuda_graph,
            )
        except Exception as exc:
            self._failed_config = config
            self.reset_feedback()
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                dreamer.release_cuda_graphs()
                gc.collect()
                torch.cuda.empty_cache()
                self.status = (
                    "CUDA memory exhausted. Lower Process Width, Pyramid Levels, "
                    "or Iterations; processing is bypassed until a setting changes."
                )
            else:
                self.status = (
                    f"Processing disabled for the current settings: {exc}"
                )
            return img01

        if (
            controls.blend > 0.0
            and previous_dream is not None
            and self.last_clean01 is not None
            and previous_dream.shape == dreamed01.shape
            and self.last_clean01.shape == clean01.shape
        ):
            dreamed01 = self._temporal.blend(
                dreamed01,
                previous_dream,
                clean01,
                self.last_clean01,
                float(controls.blend),
            )
        else:
            self._temporal.reset()

        from .gordic import restore_empty_border

        dreamed01 = restore_empty_border(clean01, dreamed01)
        self.last_dream01 = np.ascontiguousarray(dreamed01)
        self.last_clean01 = np.ascontiguousarray(clean01)
        dreamed01 = enhance_saturation(dreamed01, controls.saturation)
        self._failed_config = None
        self.status = f"OK | {self.model_name} | {layer}"
        return dreamed01


def _use_gpu_process(device: str | None) -> bool:
    """Dream outside TouchDesigner so CUDA graph capture cannot take it down."""
    if os.environ.get("DEEPDREAM_INPROCESS") == "1":
        return False
    if device == "cpu":
        return False
    forced = os.environ.get("DEEPDREAM_FORCE_ISOLATED") == "1"
    if not forced and "td" not in sys.modules:
        return False
    from .gpu_worker import worker_python

    return worker_python() is not None


class AsyncLiveDreamEngine:
    """Non-blocking facade for hosts with a real-time render/cook thread.

    Only NumPy arrays and plain Python values cross the worker boundary; no
    TouchDesigner object is ever accessed outside its main thread.
    """

    def __init__(
        self,
        model_name: str = "vgg16",
        device: str | None = None,
        layers: list[str] | None = None,
    ):
        self._engine = None
        self._worker = None
        self._future: Future[
            tuple[np.ndarray, np.ndarray, str, str, float]
        ] | None = None
        self._last_result: np.ndarray | None = None
        self._last_source: np.ndarray | None = None
        self.output_source: np.ndarray | None = None
        self.output_version = 0
        self._sync_version = -1
        self._sync_start: np.ndarray | None = None
        self._sync_target: np.ndarray | None = None
        self._sync_effect: np.ndarray | None = None
        self._sync_progress = 1.0
        self._requested_model = model_name
        self._reset_requested = False
        self.status = "Ready"
        self.dream_fps = 0.0
        self._closed = False
        self._fps_ready = False
        self._fps_signature: tuple | None = None
        self._bridge = None
        if _use_gpu_process(device):
            from .gpu_worker import GpuBridge

            try:
                self._bridge = GpuBridge()
            except Exception as exc:
                self._bridge = None
                self.status = f"GPU process unavailable: {exc}"
            else:
                self.status = self._bridge.status
                print(
                    "[DeepDreamLive] Calcolo sulla GPU in un processo separato. "
                    "Iterazioni e piramidi restano quelle del pannello; "
                    "il video continua a cucinare per conto suo."
                )
        if self._bridge is None:
            self._engine = LiveDreamEngine(model_name, device, layers)
            self._worker = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="deepdream-frame-worker"
            )

    @property
    def model_name(self) -> str:
        return self._requested_model

    def set_model(self, model_name: str, layer: str | None = None) -> None:
        if model_name not in BACKBONE_LAYERS:
            raise ValueError(f"Unknown model: {model_name}")
        self._requested_model = model_name
        self._fps_ready = False
        self._fps_signature = None
        self._last_result = None
        self._last_source = None
        self.output_source = None
        self._reset_sync()
        self.status = f"Loading {model_name} in background..."

    def reset_feedback(self) -> None:
        self._reset_requested = True
        self._last_result = None
        self._last_source = None
        self.output_source = None
        self._reset_sync()

    def _reset_sync(self) -> None:
        self._sync_version = -1
        self._sync_start = None
        self._sync_target = None
        self._sync_effect = None
        self._sync_progress = 1.0

    def _run_frame(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        proc_width: int,
        model_name: str,
        reset_feedback: bool,
        block_until_ready: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str, str, float]:
        started = time.perf_counter()
        if self._engine.model_name != model_name:
            self._engine.set_model(model_name)
        if reset_feedback:
            self._engine.reset_feedback()
        if block_until_ready and self._engine._dreamers.get(model_name) is None:
            from .backbones import DEFAULT_LAYERS

            layer = controls.layer or DEFAULT_LAYERS[model_name][0]
            error = self._engine.load_model_now(layer)
            if error:
                elapsed = time.perf_counter() - started
                return img01, img01, self._engine.status, model_name, elapsed
        result = self._engine.process_frame(
            img01, controls, proc_width=proc_width
        )
        elapsed = time.perf_counter() - started
        return result, img01, self._engine.status, model_name, elapsed

    def process_frame(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int = 416,
    ) -> np.ndarray:
        """Submit the newest frame and immediately return the latest result."""
        if self._closed:
            resized = resize_rgb01(img01, proc_width)
            self.output_source = resized
            return resized

        if self._bridge is not None:
            return self._process_isolated(img01, controls, proc_width=proc_width)

        if self._future is not None and self._future.done():
            try:
                (
                    result,
                    source,
                    status,
                    result_model,
                    elapsed,
                ) = self._future.result()
                if result_model == self._requested_model:
                    self._last_result = result
                    self._last_source = source
                    self.status = status
                    if status.startswith("OK") and elapsed > 0.0:
                        self.output_version += 1
                        measured_fps = 1.0 / elapsed
                        if self.dream_fps <= 0.0:
                            self.dream_fps = measured_fps
                        else:
                            self.dream_fps = (
                                self.dream_fps * 0.85 + measured_fps * 0.15
                            )
            except Exception as exc:
                self._last_result = None
                self._last_source = None
                self.status = f"Worker error: {exc}"
            self._future = None

        resized = resize_rgb01(img01, proc_width)
        if self._future is None:
            frame = np.ascontiguousarray(resized, dtype=np.float32).copy()
            reset = self._reset_requested
            self._reset_requested = False
            self._future = self._worker.submit(
                self._run_frame,
                frame,
                controls,
                proc_width,
                self._requested_model,
                reset,
            )

        if (
            self._last_result is not None
            and self._last_result.shape == resized.shape
        ):
            self.output_source = self._last_source
            return self._last_result
        self.output_source = resized
        return resized

    def process_frame_sync(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int = 416,
        timeout: float = 60.0,
    ) -> np.ndarray:
        """Dream this frame before returning, so the caller can lock video to it."""
        if self._closed:
            resized = resize_rgb01(img01, proc_width)
            self.output_source = resized
            return resized
        if self._bridge is not None:
            return self._process_isolated_sync(
                img01, controls, proc_width=proc_width, timeout=timeout
            )
        return self._process_thread_sync(
            img01, controls, proc_width=proc_width, timeout=timeout
        )

    def _absorb_isolated(self, finished) -> None:
        result, source, status, elapsed = finished
        self._last_result = result
        self._last_source = source
        self.status = status
        if not status.startswith("OK"):
            return
        self.output_version += 1
        if self._fps_ready and elapsed > 0.0:
            measured_fps = 1.0 / elapsed
            if self.dream_fps <= 0.0:
                self.dream_fps = measured_fps
            else:
                self.dream_fps = self.dream_fps * 0.85 + measured_fps * 0.15
        elif elapsed > 0.0:
            # The first frame at a new size also captures the CUDA graphs.
            self._fps_ready = True

    def _prepare_submit(self, img01, controls, proc_width):
        resized = resize_rgb01(img01, proc_width)
        frame = np.ascontiguousarray(resized, dtype=np.float32).copy()
        signature = (
            self._requested_model,
            controls.layer,
            int(controls.iterations),
            int(controls.pyramid),
            frame.shape,
        )
        if signature != self._fps_signature:
            self._fps_signature = signature
            self._fps_ready = False
            self.dream_fps = 0.0
        reset = self._reset_requested
        self._reset_requested = False
        return resized, frame, reset

    def _process_isolated_sync(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int,
        timeout: float,
    ) -> np.ndarray:
        deadline = time.perf_counter() + timeout
        while self._bridge.busy() and time.perf_counter() < deadline:
            finished = self._bridge.poll()
            if finished is not None:
                self._absorb_isolated(finished)
            else:
                time.sleep(0.001)
        resized, frame, reset = self._prepare_submit(img01, controls, proc_width)
        if not self._bridge.submit(
            frame, controls, proc_width, self._requested_model, reset
        ):
            self.status = self._bridge.status or "Processing failed: could not submit frame"
            self.output_source = resized
            return resized
        while time.perf_counter() < deadline:
            finished = self._bridge.poll()
            if finished is None:
                time.sleep(0.001)
                continue
            self._absorb_isolated(finished)
            if (
                self._last_result is not None
                and self._last_result.shape == resized.shape
            ):
                self.output_source = self._last_source
                return self._last_result
            self.output_source = resized
            return resized
        self.status = "Processing failed: sync timed out"
        self.output_source = resized
        if (
            self._last_result is not None
            and self._last_result.shape == resized.shape
        ):
            return self._last_result
        return resized

    def _absorb_thread(self, payload) -> None:
        result, source, status, result_model, elapsed = payload
        if result_model != self._requested_model:
            return
        self._last_result = result
        self._last_source = source
        self.status = status
        if status.startswith("OK") and elapsed > 0.0:
            self.output_version += 1
            measured_fps = 1.0 / elapsed
            if self.dream_fps <= 0.0:
                self.dream_fps = measured_fps
            else:
                self.dream_fps = self.dream_fps * 0.85 + measured_fps * 0.15

    def _process_thread_sync(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int,
        timeout: float,
    ) -> np.ndarray:
        resized = resize_rgb01(img01, proc_width)
        if self._future is not None:
            try:
                self._absorb_thread(self._future.result(timeout=timeout))
            except Exception as exc:
                self.status = f"Processing failed: sync timed out ({exc})"
                self.output_source = resized
                return resized
            self._future = None
        if self._worker is None or self._engine is None:
            self.output_source = resized
            return resized
        frame = np.ascontiguousarray(resized, dtype=np.float32).copy()
        reset = self._reset_requested
        self._reset_requested = False
        self._future = self._worker.submit(
            self._run_frame,
            frame,
            controls,
            proc_width,
            self._requested_model,
            reset,
            True,
        )
        try:
            self._absorb_thread(self._future.result(timeout=timeout))
        except Exception as exc:
            self.status = f"Processing failed: sync timed out ({exc})"
            self.output_source = resized
            return resized
        self._future = None
        if (
            self._last_result is not None
            and self._last_result.shape == resized.shape
        ):
            self.output_source = self._last_source
            return self._last_result
        self.output_source = resized
        return resized

    def composite_frame(
        self,
        current01: np.ndarray,
        dreamed01: np.ndarray,
        mask01: np.ndarray | None,
        cook_fps: float,
    ) -> np.ndarray:
        """Render a smooth live frame between slower DeepDream updates."""
        if not self.status.startswith("OK"):
            self._reset_sync()
            return current01
        source = self.output_source
        if (
            source is None
            or source.shape != dreamed01.shape
            or current01.shape != dreamed01.shape
        ):
            self._reset_sync()
            return composite_effect_delta(
                current01, dreamed01, source, mask01
            )

        target = dreamed01 - source
        if self._sync_version != self.output_version:
            if (
                self._sync_effect is None
                or self._sync_effect.shape != target.shape
            ):
                self._sync_effect = target.copy()
                self._sync_start = target.copy()
                self._sync_progress = 1.0
            else:
                self._sync_start = self._sync_effect.copy()
                self._sync_progress = 0.0
            self._sync_target = target.copy()
            self._sync_version = self.output_version

        if (
            self._sync_target is not None
            and self._sync_start is not None
            and self._sync_progress < 1.0
        ):
            jump = float(np.mean(np.abs(self._sync_target - self._sync_start)))
            if jump >= 0.12:
                self._sync_progress = 1.0
            else:
                if cook_fps > 0.0 and self.dream_fps > 0.0:
                    bridge_frames = max(
                        1, int(round(cook_fps / self.dream_fps))
                    )
                else:
                    bridge_frames = 4
                self._sync_progress = min(
                    1.0, self._sync_progress + 1.0 / bridge_frames
                )
            amount = self._sync_progress
            self._sync_effect = (
                self._sync_start * (1.0 - amount)
                + self._sync_target * amount
            )

        if self._sync_effect is None:
            self._sync_effect = target
        if mask01 is None:
            return np.clip(current01 + self._sync_effect, 0.0, 1.0)
        return np.clip(
            current01 + self._sync_effect * mask01,
            0.0,
            1.0,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        future = self._future
        if self._engine is not None:
            if future is not None and not future.done():
                future.add_done_callback(lambda _future: self._engine.close())
            else:
                self._engine.close()
        if self._worker is not None:
            self._worker.shutdown(wait=False, cancel_futures=True)
        self._last_result = None
        self._last_source = None
        self.output_source = None
        self._reset_sync()

    def _process_isolated(
        self,
        img01: np.ndarray,
        controls: DreamControls,
        *,
        proc_width: int,
    ) -> np.ndarray:
        finished = self._bridge.poll()
        if finished is not None:
            self._absorb_isolated(finished)
        elif self._bridge.status:
            self.status = self._bridge.status

        resized = resize_rgb01(img01, proc_width)
        if not self._bridge.busy():
            frame = np.ascontiguousarray(resized, dtype=np.float32).copy()
            signature = (
                self._requested_model,
                controls.layer,
                int(controls.iterations),
                int(controls.pyramid),
                frame.shape,
            )
            if signature != self._fps_signature:
                self._fps_signature = signature
                self._fps_ready = False
                self.dream_fps = 0.0
            reset = self._reset_requested
            self._reset_requested = False
            self._bridge.submit(
                frame,
                controls,
                proc_width,
                self._requested_model,
                reset,
            )

        if (
            self._last_result is not None
            and self._last_result.shape == resized.shape
        ):
            self.output_source = self._last_source
            return self._last_result
        self.output_source = resized
        return resized
