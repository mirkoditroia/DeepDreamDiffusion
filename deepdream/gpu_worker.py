"""GPU dream process.

TouchDesigner and PyTorch cannot share a CUDA context: every small kernel
pays for a switch, so a heavy pyramid falls to about one frame per second.
This process owns the context, captures one CUDA graph per resolution, and
exchanges frames through shared memory. If capture crashes, only this
process dies.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

MAX_H = 2048
MAX_W = 2048
STATE_EMPTY = 0
STATE_BUSY = 1
STATE_DONE = 2
STATE_STOP = 3
STATE_BOOT = 4

CTRL_DTYPE = np.dtype(
    [
        ("state", "<u4"),
        ("width", "<u4"),
        ("height", "<u4"),
        ("iterations", "<u4"),
        ("pyramid", "<u4"),
        ("reset", "<u4"),
        ("proc_width", "<u4"),
        ("_pad", "<u4"),
        ("lr", "<f4"),
        ("intensity", "<f4"),
        ("blend", "<f4"),
        ("feedback", "<f4"),
        ("zoom", "<f4"),
        ("rotate", "<f4"),
        ("saturation", "<f4"),
        ("elapsed", "<f4"),
        ("model", "S32"),
        ("layer", "S64"),
        ("status", "S240"),
    ],
    align=True,
)


def worker_python() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        ("touchdesigner", "venv_td", "Scripts", "python.exe"),
        ("venv_td", "Scripts", "python.exe"),
    ):
        candidate = root.joinpath(*relative)
        if candidate.is_file():
            return candidate
    return None


def _source_root(python: Path) -> Path:
    holder = python.parents[1].parent
    if (holder / "deepdream").is_dir():
        return holder
    if (holder.parent / "deepdream").is_dir():
        return holder.parent
    return Path(__file__).resolve().parents[1]


def _text(value) -> str:
    raw = bytes(value).split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace")


def _put_text(row, name: str, text: str, size: int) -> None:
    raw = text.encode("utf-8", errors="replace")[:size].ljust(size, b"\x00")
    row[name] = raw


def _process_alive(pid: int) -> bool:
    """False only when the pid is gone. Access denied still means it exists."""
    if pid <= 0:
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5


def _log_tail(path: Path, limit: int = 500) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[-limit:]


class GpuBridge:
    """Parent side. Speaks NumPy only; it never imports PyTorch."""

    def __init__(self) -> None:
        from multiprocessing import shared_memory

        python = worker_python()
        if python is None:
            raise RuntimeError("touchdesigner/venv_td/Scripts/python.exe is missing")
        self._python = python
        self._shared_memory = shared_memory
        tag = f"{os.getpid():x}{time.time_ns() & 0xFFFF:x}"
        pixels = MAX_H * MAX_W * 3 * np.dtype(np.float32).itemsize
        self._ctrl_shm = shared_memory.SharedMemory(
            name=f"ddc{tag}", create=True, size=CTRL_DTYPE.itemsize
        )
        self._src_shm = shared_memory.SharedMemory(
            name=f"dds{tag}", create=True, size=pixels
        )
        self._dst_shm = shared_memory.SharedMemory(
            name=f"ddd{tag}", create=True, size=pixels
        )
        self.ctrl = np.ndarray((1,), dtype=CTRL_DTYPE, buffer=self._ctrl_shm.buf)
        self.src = np.ndarray(
            (MAX_H, MAX_W, 3), dtype=np.float32, buffer=self._src_shm.buf
        )
        self.dst = np.ndarray(
            (MAX_H, MAX_W, 3), dtype=np.float32, buffer=self._dst_shm.buf
        )
        self.ctrl[0]["state"] = STATE_BOOT
        _put_text(self.ctrl[0], "status", "Loading GPU process...", 240)
        self.status = "Loading GPU process..."
        self._inflight: np.ndarray | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._closing = False
        self._disabled = False
        self._allow_graph = True
        self._crash_times: list[float] = []
        self._busy_since: float | None = None
        self._log_path = (
            Path(tempfile.gettempdir()) / f"deepdream_gpu_{tag}.log"
        )
        self._log = self._log_path.open("w", buffering=1, encoding="utf-8", errors="replace")
        try:
            self._spawn()
        except Exception:
            self.close()
            raise

    def busy(self) -> bool:
        if self._disabled or self.ctrl is None:
            return True
        return int(self.ctrl[0]["state"]) != STATE_EMPTY

    def poll(
        self,
    ) -> tuple[np.ndarray, np.ndarray, str, float] | None:
        self._reap_if_dead()
        if self._disabled or self.ctrl is None:
            return None
        state = int(self.ctrl[0]["state"])
        status = _text(self.ctrl[0]["status"])
        if status:
            self.status = status
        now = time.perf_counter()
        if state == STATE_BUSY:
            if self._busy_since is None:
                self._busy_since = now
            elif now - self._busy_since > 120.0:
                self._note_crash("GPU process timed out")
                self._respawn()
            return None
        self._busy_since = None
        if state == STATE_DONE and self._inflight is None:
            self.ctrl[0]["state"] = STATE_EMPTY
            return None
        if state != STATE_DONE or self._inflight is None:
            return None
        source = self._inflight
        height, width = source.shape[:2]
        result = np.empty((height, width, 3), dtype=np.float32)
        np.copyto(result, self.dst[:height, :width])
        elapsed = float(self.ctrl[0]["elapsed"])
        self._inflight = None
        self.ctrl[0]["state"] = STATE_EMPTY
        return result, source, status, elapsed

    def submit(self, frame, controls, proc_width: int, model: str, reset: bool) -> bool:
        if self._disabled or self.ctrl is None:
            return False
        if int(self.ctrl[0]["state"]) != STATE_EMPTY:
            return False
        height, width = frame.shape[:2]
        if height > MAX_H or width > MAX_W:
            self.status = "Process Width above 2048 is too large for the GPU process."
            _put_text(self.ctrl[0], "status", self.status, 240)
            return False
        np.copyto(self.src[:height, :width], frame)
        slot = self.ctrl[0]
        slot["width"] = width
        slot["height"] = height
        slot["iterations"] = max(1, int(controls.iterations))
        slot["pyramid"] = max(1, int(controls.pyramid))
        slot["reset"] = 1 if reset else 0
        slot["proc_width"] = max(1, int(proc_width))
        slot["lr"] = float(controls.lr)
        slot["intensity"] = float(controls.intensity)
        slot["blend"] = float(controls.blend)
        slot["feedback"] = float(controls.feedback)
        slot["zoom"] = float(controls.zoom)
        slot["rotate"] = float(controls.rotate)
        slot["saturation"] = float(controls.saturation)
        _put_text(slot, "model", model, 32)
        _put_text(slot, "layer", controls.layer or "", 64)
        self._inflight = frame
        self._busy_since = time.perf_counter()
        slot["state"] = STATE_BUSY
        return True

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._disabled = True
        try:
            if self.ctrl is not None:
                self.ctrl[0]["state"] = STATE_STOP
        except Exception:
            pass
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        try:
            self._log.close()
        except Exception:
            pass
        for shm in (self._ctrl_shm, self._src_shm, self._dst_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self.ctrl = None

    def _spawn(self) -> None:
        root = _source_root(self._python)
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("PYTHON"):
                env.pop(key, None)
        env["PYTHONPATH"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["DEEPDREAM_ISOLATED"] = "1"
        env["DEEPDREAM_INPROCESS"] = "1"
        torch_lib = self._python.parents[1] / "Lib" / "site-packages" / "torch" / "lib"
        env["PATH"] = (
            str(torch_lib)
            + os.pathsep
            + str(self._python.parent)
            + os.pathsep
            + env.get("PATH", "")
        )
        command = [
            str(self._python),
            "-m",
            "deepdream.gpu_worker",
            "--ctrl",
            self._ctrl_shm.name,
            "--src",
            self._src_shm.name,
            "--dst",
            self._dst_shm.name,
            "--parent-pid",
            str(os.getpid()),
        ]
        if not self._allow_graph:
            command.append("--no-graph")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._log.write(
            f"\n--- spawn graph={self._allow_graph} python={self._python} ---\n"
        )
        self._log.flush()
        self._proc = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    def _note_crash(self, message: str) -> None:
        now = time.time()
        self._crash_times = [stamp for stamp in self._crash_times if now - stamp < 30.0]
        self._crash_times.append(now)
        self._allow_graph = False
        tail = _log_tail(self._log_path)
        detail = tail or message
        self.status = f"GPU process failed: {detail}"
        _put_text(self.ctrl[0], "status", self.status, 240)
        if len(self._crash_times) >= 4:
            self._disabled = True

    def _respawn(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._inflight = None
        self._busy_since = None
        if self._disabled or self._closing:
            return
        self.ctrl[0]["state"] = STATE_BOOT
        self._spawn()

    def _reap_if_dead(self) -> None:
        proc = self._proc
        if proc is None or self._closing or proc.poll() is None:
            return
        self._note_crash(f"exit {proc.returncode}")
        self._respawn()


def _run_worker(
    ctrl_name: str, src_name: str, dst_name: str, use_graph: bool, parent_pid: int
) -> None:
    from multiprocessing import shared_memory

    os.environ["DEEPDREAM_ISOLATED"] = "1"
    import cv2
    import torch

    from .backbones import DEFAULT_LAYERS
    from .live_engine import DreamControls, LiveDreamEngine

    torch.set_num_threads(1)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    ctrl_shm = shared_memory.SharedMemory(name=ctrl_name)
    src_shm = shared_memory.SharedMemory(name=src_name)
    dst_shm = shared_memory.SharedMemory(name=dst_name)
    ctrl = np.ndarray((1,), dtype=CTRL_DTYPE, buffer=ctrl_shm.buf)
    src = np.ndarray((MAX_H, MAX_W, 3), dtype=np.float32, buffer=src_shm.buf)
    dst = np.ndarray((MAX_H, MAX_W, 3), dtype=np.float32, buffer=dst_shm.buf)
    engine = LiveDreamEngine(device="cuda")
    engine.use_cuda_graph = use_graph
    print("gpu worker online", flush=True)
    try:
        if int(ctrl[0]["state"]) == STATE_BOOT:
            ctrl[0]["state"] = STATE_EMPTY
        while True:
            state = int(ctrl[0]["state"])
            if state == STATE_STOP or not _process_alive(parent_pid):
                break
            if state != STATE_BUSY:
                time.sleep(0.001)
                continue
            started = time.perf_counter()
            status = "Processing failed"
            try:
                height = int(ctrl[0]["height"])
                width = int(ctrl[0]["width"])
                model = _text(ctrl[0]["model"]) or "vgg16"
                layer = _text(ctrl[0]["layer"]) or None
                frame = np.array(src[:height, :width], dtype=np.float32, copy=True)
                if engine.model_name != model:
                    engine.set_model(model, layer)
                active_layer = layer or DEFAULT_LAYERS[model][0]
                if engine._dreamers.get(model) is None:
                    _put_text(ctrl[0], "status", f"Loading {model}...", 240)
                    error = engine.load_model_now(active_layer)
                    if error:
                        raise RuntimeError(error)
                if int(ctrl[0]["reset"]):
                    engine.reset_feedback()
                controls = DreamControls(
                    lr=float(ctrl[0]["lr"]),
                    intensity=float(ctrl[0]["intensity"]),
                    iterations=int(ctrl[0]["iterations"]),
                    pyramid=int(ctrl[0]["pyramid"]),
                    blend=float(ctrl[0]["blend"]),
                    feedback=float(ctrl[0]["feedback"]),
                    zoom=float(ctrl[0]["zoom"]),
                    rotate=float(ctrl[0]["rotate"]),
                    saturation=float(ctrl[0]["saturation"]),
                    layer=layer,
                )
                result = engine.process_frame(
                    frame,
                    controls,
                    proc_width=int(ctrl[0]["proc_width"]),
                )
                status = engine.status
                out_h, out_w = result.shape[:2]
                dst[:out_h, :out_w] = result
                if out_h != height or out_w != width:
                    # The parent reads the submitted shape. Keep them equal.
                    dst[:height, :width] = cv2.resize(
                        result, (width, height), interpolation=cv2.INTER_CUBIC
                    )
            except Exception as exc:
                status = f"Processing failed: {exc}"
                print(status, flush=True)
                dst[: int(ctrl[0]["height"]), : int(ctrl[0]["width"])] = src[
                    : int(ctrl[0]["height"]), : int(ctrl[0]["width"])
                ]
            ctrl[0]["elapsed"] = time.perf_counter() - started
            _put_text(ctrl[0], "status", status, 240)
            ctrl[0]["state"] = STATE_DONE
    finally:
        engine.close()
        for shm in (ctrl_shm, src_shm, dst_shm):
            shm.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    _run_worker(
        args.ctrl,
        args.src,
        args.dst,
        use_graph=not args.no_graph,
        parent_pid=args.parent_pid,
    )


if __name__ == "__main__":
    main()
