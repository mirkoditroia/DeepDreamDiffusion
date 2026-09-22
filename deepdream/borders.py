"""Letterbox detection without PyTorch.

TouchDesigner imports this on the cook thread. It must not create a CUDA context.
"""

from __future__ import annotations

import numpy as np

# Nero piatto attaccato al bordo del frame (letterbox, pillarbox, fondale).
_BORDER_PEAK = 0.08
_BORDER_CONTRAST = 0.015


def _border_run(active: np.ndarray) -> tuple[int, int]:
    """Contiguous True prefix and the index where the True suffix starts."""
    length = int(active.shape[0])
    start = 0
    while start < length and bool(active[start]):
        start += 1
    end = length
    while end > start and bool(active[end - 1]):
        end -= 1
    return start, end


def empty_border_mask(img01: np.ndarray) -> np.ndarray | None:
    """Maschera delle bande nere piatte attaccate al bordo. True = pixel noto.

    Il nero di letterbox/pillarbox e' assenza di immagine. Una riga o una colonna
    conta solo se e' scura e piatta per tutta la sua lunghezza, e solo se parte
    dal bordo: un nero interno all'inquadratura resta picture.
    """
    if img01.ndim != 3 or img01.shape[2] < 3:
        return None
    rgb = np.ascontiguousarray(img01[..., :3], dtype=np.float32)
    # np.max(..., axis=2) is strangely expensive here; channel-wise maximum is not.
    peak = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    col_max = peak.max(axis=0)
    row_max = peak.max(axis=1)
    columns = (col_max < _BORDER_PEAK) & (
        (col_max - peak.min(axis=0)) < _BORDER_CONTRAST
    )
    rows = (row_max < _BORDER_PEAK) & (
        (row_max - peak.min(axis=1)) < _BORDER_CONTRAST
    )
    if not columns.any() and not rows.any():
        return None

    height, width = peak.shape
    left, right = _border_run(columns)
    top, bottom = _border_run(rows)
    if left == 0 and right == width and top == 0 and bottom == height:
        return None

    void = np.zeros((height, width), dtype=bool)
    if left:
        void[:, :left] = True
    if right < width:
        void[:, right:] = True
    if top:
        void[:top, :] = True
    if bottom < height:
        void[bottom:, :] = True
    if not void.any():
        return None
    return void


def restore_empty_border(clean01: np.ndarray, image01: np.ndarray) -> np.ndarray:
    """Ricopia il nero di bordo dal frame sorgente. Il blend temporale non puo' ridipingerlo."""
    if clean01.shape != image01.shape:
        return image01
    void = empty_border_mask(clean01)
    if void is None:
        return image01
    return np.where(void[..., None], clean01, image01)
