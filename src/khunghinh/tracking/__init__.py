"""Tracking đa đối tượng nhẹ (ByteTrack-lite) — gán track_id ổn định cho khuôn mặt."""
from __future__ import annotations

from .iou_tracker import IouTracker, Track, iou

__all__ = ["IouTracker", "Track", "iou"]
