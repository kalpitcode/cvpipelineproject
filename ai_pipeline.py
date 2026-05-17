from __future__ import annotations

import argparse
import importlib
import json
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import numpy as np


DEFAULT_OUTPUT = Path(__file__).with_name("ai_pipeline_output.mp4")
DEFAULT_RUNS_DIR = Path(__file__).with_name("surveillance_runs")
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")
DISTRESS_EMOTIONS = {"angry", "fear", "sad", "disgust"}
PHONE_CLASS_NAMES = {"cell phone", "mobile phone"}
POSE_CONNECTION_COLOR = (0, 255, 0)


@dataclass
class PoseBundle:
    pose: Any
    drawing_utils: Any
    drawing_styles: Any
    pose_module: Any
    mediapipe: Any
    api: str
    connections: Any


@dataclass
class CaptionBundle:
    processor: Any
    model: Any
    torch: Any
    pil_image: Any


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    previous_centroid: tuple[int, int]
    first_seen: float
    last_seen: float
    misses: int = 0
    loiter_alerted: bool = False
    seen_in_frame: bool = True
    last_emotion: str = "unknown"
    fall_alerted_at: float = 0.0
    intrusion_alerted_at: float = 0.0
    polygon_alerted_at: float = 0.0
    inside_polygon: bool = False

    def dwell_seconds(self, now: float) -> float:
        return max(0.0, now - self.first_seen)


@dataclass
class EventRecord:
    timestamp: str
    event_type: str
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_path: str | None = None


@dataclass
class SessionPaths:
    root: Path
    snapshot_dir: Path
    event_log_path: Path
    summary_path: Path


class AsyncCaptioner:
    def __init__(self, bundle: CaptionBundle):
        self.bundle = bundle
        self._queue: queue.Queue[tuple[int, Any] | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._lock = threading.Lock()
        self._latest_caption = "Caption warming up..."
        self._latest_latency_ms = 0.0
        self._latest_frame_index = -1
        self._latest_error: str | None = None
        self._thread.start()

    def submit(self, frame_index: int, rgb_frame: Any) -> bool:
        payload = (frame_index, rgb_frame.copy())
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def latest(self) -> tuple[str, float, str | None]:
        with self._lock:
            return self._latest_caption, self._latest_latency_ms, self._latest_error

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            frame_index, rgb_frame = item
            started = time.perf_counter()
            try:
                caption = generate_caption(self.bundle, rgb_frame)
                latency_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._latest_caption = caption or self._latest_caption
                    self._latest_latency_ms = latency_ms
                    self._latest_frame_index = frame_index
                    self._latest_error = None
            except Exception as exc:
                with self._lock:
                    self._latest_error = str(exc)
            finally:
                self._queue.task_done()


class SurveillanceLogger:
    def __init__(self, session_paths: SessionPaths, snapshot_cooldown: float):
        self.session_paths = session_paths
        self.snapshot_cooldown = snapshot_cooldown
        self.event_counts: dict[str, int] = {}
        self.recent_events: deque[EventRecord] = deque(maxlen=8)
        self._snapshot_times: dict[str, float] = {}

    def record(
        self,
        event_type: str,
        severity: str,
        message: str,
        now: float,
        metadata: dict[str, Any] | None = None,
        frame: Any | None = None,
        snapshot_key: str | None = None,
    ) -> EventRecord:
        metadata = metadata or {}
        snapshot_path = None
        if frame is not None and snapshot_key is not None and self._should_capture(snapshot_key, now):
            snapshot_path = self._write_snapshot(frame, event_type, now)

        record = EventRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=metadata,
            snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
        )
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        self.recent_events.appendleft(record)
        with self.session_paths.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")
        return record

    def write_summary(
        self,
        runtime_seconds: float,
        frames_processed: int,
        average_fps: float,
        max_people: int,
        latest_caption: str,
        output_video_path: str | None,
    ) -> None:
        summary = {
            "runtime_seconds": round(runtime_seconds, 2),
            "frames_processed": frames_processed,
            "average_fps": round(average_fps, 2),
            "max_people_observed": max_people,
            "event_counts": self.event_counts,
            "latest_caption": latest_caption,
            "output_video_path": output_video_path,
            "event_log_path": str(self.session_paths.event_log_path),
            "snapshot_dir": str(self.session_paths.snapshot_dir),
        }
        self.session_paths.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _should_capture(self, snapshot_key: str, now: float) -> bool:
        last = self._snapshot_times.get(snapshot_key)
        if last is not None and now - last < self.snapshot_cooldown:
            return False
        self._snapshot_times[snapshot_key] = now
        return True

    def _write_snapshot(self, frame: Any, event_type: str, now: float) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        snapshot_path = self.session_paths.snapshot_dir / f"{timestamp}_{event_type}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return snapshot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI Smart Surveillance System: real-time CPU-only surveillance analytics "
            "with YOLOv8, FER, MediaPipe Pose, and BLIP."
        ),
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--input-video", type=Path, default=None, help="Optional input video file to analyze instead of a live webcam.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--caption-interval", type=int, default=30, help="Generate a BLIP caption every N frames.")
    parser.add_argument("--track-max-missed", type=int, default=20, help="Frames a person track can disappear before removal.")
    parser.add_argument("--track-distance", type=int, default=140, help="Maximum centroid distance for track matching.")
    parser.add_argument("--loiter-seconds", type=float, default=10.0, help="Seconds before a visible person triggers loitering.")
    parser.add_argument("--no-person-alert-seconds", type=float, default=15.0, help="Seconds with no visible people before logging an empty-scene alert.")
    parser.add_argument("--crowd-threshold", type=int, default=3, help="Visible person count that triggers a crowd alert.")
    parser.add_argument("--distress-threshold", type=float, default=0.72, help="Minimum FER confidence for a distress-emotion alert.")
    parser.add_argument("--snapshot-cooldown", type=float, default=8.0, help="Seconds between evidence snapshots for the same alert type.")
    parser.add_argument("--intrusion-line-y", type=float, default=0.55, help="Normalized vertical position of a horizontal intrusion line from 0.0 to 1.0.")
    parser.add_argument("--intrusion-line-x", type=float, default=0.5, help="Normalized horizontal position of a vertical intrusion line from 0.0 to 1.0.")
    parser.add_argument("--line-orientation", choices=("horizontal", "vertical"), default="horizontal", help="Whether the tripwire is horizontal or vertical.")
    parser.add_argument("--line-direction", choices=("down", "up", "left", "right", "any"), default="down", help="Which crossing direction should trigger an intrusion alert.")
    parser.add_argument("--restricted-polygon", type=str, default="", help="Normalized polygon points like '0.1,0.2;0.8,0.2;0.8,0.7;0.1,0.7'.")
    parser.add_argument("--fall-angle-threshold", type=float, default=35.0, help="Maximum torso angle from horizontal before a pose is considered fall-like.")
    parser.add_argument("--fall-aspect-threshold", type=float, default=1.1, help="Minimum pose box aspect ratio width/height for fall detection.")
    parser.add_argument("--no-line-crossing", action="store_true", help="Disable line-crossing intrusion detection.")
    parser.add_argument("--no-polygon-zone", action="store_true", help="Disable polygon restricted-zone detection.")
    parser.add_argument("--no-fall", action="store_true", help="Disable pose-based fall detection.")
    parser.add_argument("--session-name", type=str, default=None, help="Optional run folder name inside the surveillance log directory.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Directory where event logs, summaries, and snapshots are stored.")
    parser.add_argument("--no-emotion", action="store_true", help="Disable FER facial emotion recognition.")
    parser.add_argument("--no-pose", action="store_true", help="Disable MediaPipe pose estimation.")
    parser.add_argument("--no-caption", action="store_true", help="Disable BLIP scene captioning.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening the OpenCV preview window.")
    parser.add_argument("--save", action="store_true", help="Save the rendered output video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path when --save is used. Default: {DEFAULT_OUTPUT.name}",
    )
    return parser


def require_dependency(import_name: str, pip_name: str) -> Any:
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        message = str(exc)
        if "No module named" in message:
            raise RuntimeError(
                f"Missing dependency '{pip_name}'. Install it with: pip install {pip_name}"
            ) from exc
        raise RuntimeError(
            f"Dependency '{pip_name}' is installed but failed to import: {message}"
        ) from exc


def load_yolo_model() -> Any:
    ultralytics = require_dependency("ultralytics", "ultralytics")
    model = ultralytics.YOLO("yolov8n.pt")
    model.to("cpu")
    return model


def load_emotion_detector() -> Any:
    try:
        fer_module = importlib.import_module("fer.fer")
    except ImportError as exc:
        message = str(exc)
        if "No module named" in message:
            raise RuntimeError(
                f"FER is installed but one of its internal dependencies is missing: {message}"
            ) from exc
        raise RuntimeError(f"FER failed to import: {message}") from exc
    return fer_module.FER(mtcnn=False)


def ensure_pose_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    try:
        urlretrieve(POSE_MODEL_URL, model_path)
    except URLError as exc:
        raise RuntimeError(
            "MediaPipe Pose model download failed. Check your internet connection "
            f"or download the model manually to '{model_path.name}'."
        ) from exc
    return model_path


def load_pose_bundle() -> PoseBundle:
    mediapipe = require_dependency("mediapipe", "mediapipe")
    if hasattr(mediapipe, "solutions"):
        pose_module = mediapipe.solutions.pose
        pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return PoseBundle(
            pose=pose,
            drawing_utils=mediapipe.solutions.drawing_utils,
            drawing_styles=mediapipe.solutions.drawing_styles,
            pose_module=pose_module,
            mediapipe=mediapipe,
            api="solutions",
            connections=pose_module.POSE_CONNECTIONS,
        )

    try:
        drawing_utils = importlib.import_module("mediapipe.tasks.python.vision.drawing_utils")
        drawing_styles = importlib.import_module("mediapipe.tasks.python.vision.drawing_styles")
        pose_module = importlib.import_module("mediapipe.tasks.python.vision.pose_landmarker")
    except ImportError as exc:
        raise RuntimeError(f"MediaPipe pose components failed to import: {exc}") from exc

    model_path = ensure_pose_model(POSE_MODEL_PATH)
    base_options = mediapipe.tasks.BaseOptions(model_asset_path=str(model_path))
    pose_options = mediapipe.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose = mediapipe.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    return PoseBundle(
        pose=pose,
        drawing_utils=drawing_utils,
        drawing_styles=drawing_styles,
        pose_module=pose_module,
        mediapipe=mediapipe,
        api="tasks",
        connections=pose_module.PoseLandmarksConnections.POSE_LANDMARKS,
    )


def load_caption_bundle() -> CaptionBundle:
    torch = require_dependency("torch", "torch")
    transformers = require_dependency("transformers", "transformers")
    pil_image = require_dependency("PIL.Image", "Pillow")

    processor = transformers.BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = transformers.BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.to("cpu")
    model.eval()
    return CaptionBundle(processor=processor, model=model, torch=torch, pil_image=pil_image)


def build_session_paths(log_dir: Path, session_name: str | None) -> SessionPaths:
    session_id = session_name or time.strftime("run_%Y%m%d_%H%M%S")
    root = log_dir / session_id
    snapshot_dir = root / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return SessionPaths(
        root=root,
        snapshot_dir=snapshot_dir,
        event_log_path=root / "events.jsonl",
        summary_path=root / "session_summary.json",
    )


def compute_fps(previous_time: float) -> tuple[float, float]:
    now = time.perf_counter()
    elapsed = now - previous_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    return fps, now


def init_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def generate_caption(bundle: CaptionBundle, rgb_frame: Any) -> str:
    image = bundle.pil_image.fromarray(rgb_frame)
    image.thumbnail((640, 640))
    inputs = bundle.processor(images=image, return_tensors="pt")
    inputs = {key: value.to("cpu") for key, value in inputs.items()}
    with bundle.torch.inference_mode():
        output = bundle.model.generate(**inputs, max_new_tokens=20)
    return bundle.processor.decode(output[0], skip_special_tokens=True).strip()


def parse_yolo_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = str(names[cls_id])
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": confidence,
                "centroid": ((x1 + x2) // 2, (y1 + y2) // 2),
            }
        )
    return detections


def centroid_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou_xyxy(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


def update_tracks(
    tracks: dict[int, TrackState],
    next_track_id: int,
    person_detections: list[dict[str, Any]],
    now: float,
    max_distance: float,
    max_missed: int,
) -> tuple[dict[int, TrackState], int, list[EventRecord], dict[int, dict[str, Any]]]:
    events: list[EventRecord] = []
    detection_by_track: dict[int, dict[str, Any]] = {}
    candidates: list[tuple[float, int, int]] = []
    track_ids = list(tracks.keys())

    for track_id in track_ids:
        tracks[track_id].seen_in_frame = False
        for detection_index, detection in enumerate(person_detections):
            distance = centroid_distance(tracks[track_id].centroid, detection["centroid"])
            if distance <= max_distance:
                candidates.append((distance, track_id, detection_index))

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for _, track_id, detection_index in sorted(candidates, key=lambda item: item[0]):
        if track_id in matched_tracks or detection_index in matched_detections:
            continue

        detection = person_detections[detection_index]
        track = tracks[track_id]
        track.previous_centroid = track.centroid
        track.bbox = detection["bbox"]
        track.centroid = detection["centroid"]
        track.last_seen = now
        track.misses = 0
        track.seen_in_frame = True
        matched_tracks.add(track_id)
        matched_detections.add(detection_index)
        detection_by_track[track_id] = detection

    for detection_index, detection in enumerate(person_detections):
        if detection_index in matched_detections:
            continue
        track = TrackState(
            track_id=next_track_id,
            bbox=detection["bbox"],
            centroid=detection["centroid"],
            previous_centroid=detection["centroid"],
            first_seen=now,
            last_seen=now,
        )
        tracks[next_track_id] = track
        detection_by_track[next_track_id] = detection
        events.append(
            EventRecord(
                timestamp="",
                event_type="person_entered",
                severity="info",
                message=f"Person #{next_track_id} entered the scene.",
                metadata={"track_id": next_track_id, "bbox": list(detection["bbox"])},
            )
        )
        next_track_id += 1

    removed_track_ids: list[int] = []
    for track_id, track in tracks.items():
        if track_id in matched_tracks or track_id in detection_by_track:
            continue
        track.misses += 1
        if track.misses > max_missed:
            removed_track_ids.append(track_id)
            events.append(
                EventRecord(
                    timestamp="",
                    event_type="person_left",
                    severity="info",
                    message=f"Person #{track_id} left after {track.dwell_seconds(now):.1f}s.",
                    metadata={"track_id": track_id, "dwell_seconds": round(track.dwell_seconds(now), 2)},
                )
            )

    for track_id in removed_track_ids:
        del tracks[track_id]

    return tracks, next_track_id, events, detection_by_track


def find_track_for_face(
    face_box: tuple[int, int, int, int],
    active_tracks: dict[int, TrackState],
) -> int | None:
    best_track_id = None
    best_iou = 0.0
    fx, fy, fw, fh = face_box
    face_xyxy = (fx, fy, fx + fw, fy + fh)
    for track_id, track in active_tracks.items():
        if not track.seen_in_frame:
            continue
        overlap = iou_xyxy(face_xyxy, track.bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_track_id = track_id
    return best_track_id if best_iou > 0 else None


def draw_label(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (30, 30, 30),
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 6
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - padding),
        (x + text_w + (padding * 2), y + padding // 2),
        bg_color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x + padding, y - baseline),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_top_hud(frame: Any, fps: float, timings_ms: dict[str, float], people_count: int, active_alerts: int) -> None:
    draw_label(frame, f"FPS {fps:5.1f}", (12, 30), bg_color=(22, 22, 22))
    timing_text = (
        f"YOLO {timings_ms['yolo']:.0f}ms | "
        f"FER {timings_ms['emotion']:.0f}ms | "
        f"POSE {timings_ms['pose']:.0f}ms | "
        f"CAP {timings_ms['caption']:.0f}ms"
    )
    draw_label(frame, timing_text, (12, 58), bg_color=(22, 22, 22))
    draw_label(frame, f"People {people_count} | Alerts {active_alerts}", (12, 86), bg_color=(22, 22, 22))


def draw_intrusion_line(frame: Any, line_value_px: int, direction: str, orientation: str) -> None:
    height, width = frame.shape[:2]
    if orientation == "horizontal":
        line_y_px = max(0, min(height - 1, line_value_px))
        cv2.line(frame, (0, line_y_px), (width, line_y_px), (0, 80, 255), 2, cv2.LINE_AA)
        draw_label(frame, f"Restricted line ({direction})", (12, max(120, line_y_px - 8)), bg_color=(0, 80, 255))
        return
    line_x_px = max(0, min(width - 1, line_value_px))
    cv2.line(frame, (line_x_px, 0), (line_x_px, height), (0, 80, 255), 2, cv2.LINE_AA)
    draw_label(frame, f"Restricted line ({direction})", (max(12, line_x_px + 8), 120), bg_color=(0, 80, 255))


def draw_restricted_polygon(frame: Any, polygon_points: list[tuple[int, int]]) -> None:
    if len(polygon_points) < 3:
        return
    overlay = frame.copy()
    contour = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(overlay, [contour], color=(30, 30, 180))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [contour], isClosed=True, color=(40, 40, 255), thickness=2, lineType=cv2.LINE_AA)
    min_x = int(contour[:, 0].min())
    min_y = int(contour[:, 1].min())
    draw_label(frame, "Restricted zone", (max(12, min_x), max(24, min_y + 24)), bg_color=(40, 40, 255))


def draw_caption_bar(frame: Any, caption: str) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    bar_height = 76
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (12, 12, 12), thickness=-1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = 0.65
    prefix = "Scene:"
    cv2.putText(
        frame,
        prefix,
        (16, height - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (125, 235, 160),
        2,
        cv2.LINE_AA,
    )
    available_width = width - 135
    for index, line in enumerate(wrap_text(caption or "Caption warming up...", available_width, font_scale, 1)):
        cv2.putText(
            frame,
            line,
            (118, height - 44 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def draw_event_feed(frame: Any, recent_events: deque[EventRecord]) -> None:
    if not recent_events:
        return

    width = frame.shape[1]
    x = max(12, width - 420)
    y = 30
    draw_label(frame, "Recent Events", (x, y), bg_color=(35, 35, 35))
    line_y = y + 28
    color_map = {"info": (70, 160, 255), "warn": (0, 191, 255), "critical": (0, 80, 255)}
    for record in list(recent_events)[:5]:
        bg = color_map.get(record.severity, (60, 60, 60))
        text = f"[{record.timestamp[-8:]}] {record.event_type}: {record.message}"
        draw_label(frame, text[:48], (x, line_y), bg_color=bg)
        line_y += 28


def draw_other_detections(frame: Any, detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        color = (40, 190, 250)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, (x1, max(y1, 20)), bg_color=color, text_color=(10, 10, 10))


def draw_person_tracks(frame: Any, active_tracks: dict[int, TrackState], now: float) -> None:
    for track in active_tracks.values():
        if not track.seen_in_frame:
            continue
        x1, y1, x2, y2 = track.bbox
        color = (80, 230, 120) if not track.loiter_alerted else (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tags: list[str] = []
        if track.intrusion_alerted_at > 0:
            tags.append("intrusion")
        if track.inside_polygon:
            tags.append("zone")
        if track.fall_alerted_at > 0:
            tags.append("fall")
        suffix = f" | {'/'.join(tags)}" if tags else ""
        label = f"ID {track.track_id} | {track.dwell_seconds(now):.1f}s | {track.last_emotion}{suffix}"
        draw_label(frame, label, (x1, max(y1, 20)), bg_color=color, text_color=(10, 10, 10))


def draw_emotion_annotations(
    frame: Any,
    detections: list[dict[str, Any]],
    active_tracks: dict[int, TrackState],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for detection in detections:
        emotions = detection.get("emotions") or {}
        if not emotions:
            continue

        x, y, w, h = (int(value) for value in detection["box"])
        emotion, score = max(emotions.items(), key=lambda item: item[1])
        track_id = find_track_for_face((x, y, w, h), active_tracks)
        if track_id is not None and track_id in active_tracks:
            active_tracks[track_id].last_emotion = emotion

        label_prefix = f"ID {track_id} " if track_id is not None else ""
        label = f"{label_prefix}{emotion} {score:.2f}"
        color = (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, label, (x, max(y, 20)), bg_color=color, text_color=(20, 20, 20))
        summaries.append({"emotion": emotion, "score": float(score), "track_id": track_id, "face_box": (x, y, w, h)})
    return summaries


def draw_pose(frame: Any, pose_bundle: PoseBundle, pose_result: Any) -> None:
    if pose_bundle.api == "solutions":
        if not pose_result.pose_landmarks:
            return
        pose_bundle.drawing_utils.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            pose_bundle.connections,
            landmark_drawing_spec=pose_bundle.drawing_styles.get_default_pose_landmarks_style(),
        )
        return

    if not pose_result.pose_landmarks:
        return

    landmark_style = pose_bundle.drawing_styles.get_default_pose_landmarks_style()
    connection_style = pose_bundle.drawing_utils.DrawingSpec(color=POSE_CONNECTION_COLOR, thickness=2)
    for pose_landmarks in pose_result.pose_landmarks:
        pose_bundle.drawing_utils.draw_landmarks(
            frame,
            pose_landmarks,
            pose_bundle.connections,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )


def wrap_text(text: str, max_width: int, font_scale: float, thickness: int) -> list[str]:
    if not text:
        return [""]
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def build_pose_result(pose_bundle: PoseBundle, rgb_frame: Any) -> Any:
    if pose_bundle.api == "solutions":
        return pose_bundle.pose.process(rgb_frame)
    mp_image = pose_bundle.mediapipe.Image(
        image_format=pose_bundle.mediapipe.ImageFormat.SRGB,
        data=rgb_frame,
    )
    timestamp_ms = int(time.perf_counter() * 1000)
    return pose_bundle.pose.detect_for_video(mp_image, timestamp_ms)


def parse_normalized_polygon(raw_value: str) -> list[tuple[float, float]]:
    if not raw_value.strip():
        return []
    points: list[tuple[float, float]] = []
    for pair in raw_value.split(";"):
        cleaned = pair.strip()
        if not cleaned:
            continue
        try:
            x_text, y_text = cleaned.split(",")
            x = float(x_text)
            y = float(y_text)
        except ValueError as exc:
            raise RuntimeError(
                "Invalid --restricted-polygon format. Use 'x1,y1;x2,y2;x3,y3'."
            ) from exc
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise RuntimeError("Restricted polygon coordinates must stay between 0.0 and 1.0.")
        points.append((x, y))
    if points and len(points) < 3:
        raise RuntimeError("Restricted polygon needs at least 3 points.")
    return points


def polygon_to_pixels(
    normalized_points: list[tuple[float, float]],
    frame_shape: tuple[int, int, int],
) -> list[tuple[int, int]]:
    height, width = frame_shape[:2]
    return [
        (int(x * width), int(y * height))
        for x, y in normalized_points
    ]


def line_crossed(
    previous_centroid: tuple[int, int],
    current_centroid: tuple[int, int],
    line_value: int,
    direction: str,
    orientation: str,
) -> bool:
    if orientation == "horizontal":
        previous_axis = previous_centroid[1]
        current_axis = current_centroid[1]
        if direction == "down":
            return previous_axis < line_value <= current_axis
        if direction == "up":
            return previous_axis > line_value >= current_axis
        crossed_down = previous_axis < line_value <= current_axis
        crossed_up = previous_axis > line_value >= current_axis
        return crossed_down or crossed_up

    previous_axis = previous_centroid[0]
    current_axis = current_centroid[0]
    if direction == "right":
        return previous_axis < line_value <= current_axis
    if direction == "left":
        return previous_axis > line_value >= current_axis
    crossed_right = previous_axis < line_value <= current_axis
    crossed_left = previous_axis > line_value >= current_axis
    return crossed_right or crossed_left


def detect_line_crossing_events(
    tracks: dict[int, TrackState],
    line_value: int,
    direction: str,
    orientation: str,
    now: float,
    cooldown_seconds: float,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    for track in tracks.values():
        if not track.seen_in_frame:
            continue
        if line_crossed(track.previous_centroid, track.centroid, line_value, direction, orientation) and now - track.intrusion_alerted_at >= cooldown_seconds:
            track.intrusion_alerted_at = now
            events.append(
                EventRecord(
                    timestamp="",
                    event_type="line_crossing_intrusion",
                    severity="critical",
                    message=f"Person #{track.track_id} crossed the restricted line.",
                    metadata={
                        "track_id": track.track_id,
                        "previous_centroid": list(track.previous_centroid),
                        "current_centroid": list(track.centroid),
                        "line_value": line_value,
                        "line_orientation": orientation,
                        "direction": direction,
                    },
                )
            )
    return events


def detect_polygon_zone_events(
    tracks: dict[int, TrackState],
    polygon_points: list[tuple[int, int]],
    now: float,
    cooldown_seconds: float,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    if len(polygon_points) < 3:
        return events

    contour = np.array(polygon_points, dtype=np.int32)
    for track in tracks.values():
        if not track.seen_in_frame:
            track.inside_polygon = False
            continue

        inside_now = cv2.pointPolygonTest(contour, track.centroid, False) >= 0
        if inside_now and not track.inside_polygon and now - track.polygon_alerted_at >= cooldown_seconds:
            track.polygon_alerted_at = now
            events.append(
                EventRecord(
                    timestamp="",
                    event_type="restricted_zone_entry",
                    severity="critical",
                    message=f"Person #{track.track_id} entered the restricted zone.",
                    metadata={
                        "track_id": track.track_id,
                        "centroid": list(track.centroid),
                    },
                )
            )
        track.inside_polygon = inside_now
    return events


def pose_bbox_from_landmarks(landmarks: list[Any], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    xs = [landmark.x for landmark in landmarks if 0.0 <= landmark.x <= 1.0]
    ys = [landmark.y for landmark in landmarks if 0.0 <= landmark.y <= 1.0]
    if not xs or not ys:
        return None
    x1 = int(min(xs) * width)
    y1 = int(min(ys) * height)
    x2 = int(max(xs) * width)
    y2 = int(max(ys) * height)
    return x1, y1, x2, y2


def match_pose_to_track(pose_box: tuple[int, int, int, int], tracks: dict[int, TrackState]) -> int | None:
    best_track_id = None
    best_overlap = 0.0
    for track_id, track in tracks.items():
        if not track.seen_in_frame:
            continue
        overlap = iou_xyxy(pose_box, track.bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best_track_id = track_id
    return best_track_id if best_overlap > 0.05 else None


def extract_pose_landmark_sets(pose_bundle: PoseBundle, pose_result: Any) -> list[list[Any]]:
    if pose_bundle.api == "solutions":
        if not pose_result.pose_landmarks:
            return []
        return [list(pose_result.pose_landmarks.landmark)]
    return list(pose_result.pose_landmarks or [])


def detect_fall_events(
    pose_bundle: PoseBundle,
    pose_result: Any,
    tracks: dict[int, TrackState],
    frame_shape: tuple[int, int, int],
    now: float,
    angle_threshold: float,
    aspect_threshold: float,
    cooldown_seconds: float,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    pose_sets = extract_pose_landmark_sets(pose_bundle, pose_result)
    left_shoulder = int(pose_bundle.pose_module.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = int(pose_bundle.pose_module.PoseLandmark.RIGHT_SHOULDER)
    left_hip = int(pose_bundle.pose_module.PoseLandmark.LEFT_HIP)
    right_hip = int(pose_bundle.pose_module.PoseLandmark.RIGHT_HIP)

    for landmarks in pose_sets:
        if len(landmarks) <= max(left_shoulder, right_shoulder, left_hip, right_hip):
            continue
        pose_box = pose_bbox_from_landmarks(landmarks, frame_shape)
        if pose_box is None:
            continue

        ls = landmarks[left_shoulder]
        rs = landmarks[right_shoulder]
        lh = landmarks[left_hip]
        rh = landmarks[right_hip]
        shoulder_center = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
        hip_center = ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
        dx = abs(hip_center[0] - shoulder_center[0])
        dy = abs(hip_center[1] - shoulder_center[1])
        torso_angle = math.degrees(math.atan2(dy, dx + 1e-6))
        x1, y1, x2, y2 = pose_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = width / float(height)
        track_id = match_pose_to_track(pose_box, tracks)
        if track_id is None:
            continue

        track = tracks[track_id]
        if torso_angle <= angle_threshold and aspect_ratio >= aspect_threshold and now - track.fall_alerted_at >= cooldown_seconds:
            track.fall_alerted_at = now
            events.append(
                EventRecord(
                    timestamp="",
                    event_type="fall_detected",
                    severity="critical",
                    message=f"Potential fall detected for person #{track_id}.",
                    metadata={
                        "track_id": track_id,
                        "torso_angle": round(torso_angle, 2),
                        "aspect_ratio": round(aspect_ratio, 2),
                        "pose_box": list(pose_box),
                    },
                )
            )
    return events


def emit_runtime_events(
    logger: SurveillanceLogger,
    pending_events: list[EventRecord],
    now: float,
    frame: Any,
) -> int:
    emitted = 0
    for pending in pending_events:
        pending.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        logger.record(
            pending.event_type,
            pending.severity,
            pending.message,
            now,
            metadata=pending.metadata,
            frame=frame,
            snapshot_key=pending.event_type,
        )
        emitted += 1
    return emitted


def main() -> int:
    args = build_parser().parse_args()
    if args.input_video is not None and not args.input_video.exists():
        print(f"Input video not found: {args.input_video}", file=sys.stderr)
        return 1

    try:
        normalized_polygon = parse_normalized_polygon(args.restricted_polygon)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.line_orientation == "horizontal" and args.line_direction in {"left", "right"}:
        print("Horizontal lines only support directions: down, up, any.", file=sys.stderr)
        return 1
    if args.line_orientation == "vertical" and args.line_direction in {"down", "up"}:
        print("Vertical lines only support directions: left, right, any.", file=sys.stderr)
        return 1

    session_paths = build_session_paths(args.log_dir, args.session_name)
    logger = SurveillanceLogger(session_paths, snapshot_cooldown=args.snapshot_cooldown)

    try:
        yolo_model = load_yolo_model()
        emotion_detector = None if args.no_emotion else load_emotion_detector()
        pose_bundle = None if args.no_pose else load_pose_bundle()
        caption_bundle = None if args.no_caption else load_caption_bundle()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    captioner = None if caption_bundle is None else AsyncCaptioner(caption_bundle)
    if args.input_video is not None:
        capture = cv2.VideoCapture(str(args.input_video))
        source_label = str(args.input_video)
    else:
        capture = cv2.VideoCapture(args.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source_label = f"camera:{args.camera_index}"

    if not capture.isOpened():
        if args.input_video is not None:
            print(f"Unable to open input video: {args.input_video}", file=sys.stderr)
        else:
            print(f"Unable to open webcam index {args.camera_index}.", file=sys.stderr)
        return 1

    show_window = not args.no_display

    writer = None
    frame_index = 0
    previous_time = time.perf_counter()
    session_started_perf = previous_time
    session_started_wall = time.time()
    average_fps_accumulator = 0.0
    active_alerts = 0
    visible_people_max = 0

    tracks: dict[int, TrackState] = {}
    next_track_id = 1
    no_person_started_at: float | None = None
    no_person_alert_active = False
    crowd_alert_active = False
    last_phone_alert_at = 0.0
    last_distress_alert_at = 0.0
    line_value_px: int | None = None
    polygon_points_px: list[tuple[int, int]] = []

    print(f"Session artifacts: {session_paths.root}")
    logger.record(
        "session_started",
        "info",
        "Surveillance session started.",
        session_started_wall,
        metadata={
            "camera_index": args.camera_index if args.input_video is None else None,
            "input_video": str(args.input_video) if args.input_video is not None else None,
            "source": source_label,
            "save_output": args.save,
            "display_enabled": show_window,
        },
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Webcam frame read failed.", file=sys.stderr)
                break

            if writer is None and args.save:
                capture_fps = capture.get(cv2.CAP_PROP_FPS)
                output_fps = capture_fps if capture_fps and capture_fps > 1 else 20.0
                frame_height, frame_width = frame.shape[:2]
                writer = init_video_writer(args.output, output_fps, (frame_width, frame_height))
            if line_value_px is None:
                if args.line_orientation == "horizontal":
                    line_value_px = int(frame.shape[0] * max(0.0, min(1.0, args.intrusion_line_y)))
                else:
                    line_value_px = int(frame.shape[1] * max(0.0, min(1.0, args.intrusion_line_x)))
            if normalized_polygon and not polygon_points_px:
                polygon_points_px = polygon_to_pixels(normalized_polygon, frame.shape)

            now = time.time()
            timings_ms = {"yolo": 0.0, "emotion": 0.0, "pose": 0.0, "caption": 0.0}
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yolo_started = time.perf_counter()
            yolo_result = yolo_model.predict(
                source=frame,
                conf=args.yolo_conf,
                device="cpu",
                verbose=False,
            )[0]
            detections = parse_yolo_detections(yolo_result)
            timings_ms["yolo"] = (time.perf_counter() - yolo_started) * 1000.0

            person_detections = [d for d in detections if d["class_name"] == "person"]
            other_detections = [d for d in detections if d["class_name"] != "person"]

            tracks, next_track_id, track_events, _ = update_tracks(
                tracks,
                next_track_id,
                person_detections,
                now,
                max_distance=float(args.track_distance),
                max_missed=args.track_max_missed,
            )
            emitted_events = emit_runtime_events(logger, track_events, now, frame)

            visible_people = sum(1 for track in tracks.values() if track.seen_in_frame)
            visible_people_max = max(visible_people_max, visible_people)

            if not args.no_line_crossing and line_value_px is not None:
                line_events = detect_line_crossing_events(
                    tracks,
                    line_value=line_value_px,
                    direction=args.line_direction,
                    orientation=args.line_orientation,
                    now=now,
                    cooldown_seconds=args.snapshot_cooldown,
                )
                emitted_events += emit_runtime_events(logger, line_events, now, frame)
            if not args.no_polygon_zone and polygon_points_px:
                polygon_events = detect_polygon_zone_events(
                    tracks,
                    polygon_points=polygon_points_px,
                    now=now,
                    cooldown_seconds=args.snapshot_cooldown,
                )
                emitted_events += emit_runtime_events(logger, polygon_events, now, frame)

            for track in tracks.values():
                if track.seen_in_frame and not track.loiter_alerted and track.dwell_seconds(now) >= args.loiter_seconds:
                    track.loiter_alerted = True
                    logger.record(
                        "loitering_detected",
                        "warn",
                        f"Person #{track.track_id} has remained in view for {track.dwell_seconds(now):.1f}s.",
                        now,
                        metadata={"track_id": track.track_id, "dwell_seconds": round(track.dwell_seconds(now), 2)},
                        frame=frame,
                        snapshot_key=f"loitering_{track.track_id}",
                    )
                    emitted_events += 1

            if visible_people == 0:
                if no_person_started_at is None:
                    no_person_started_at = now
                if not no_person_alert_active and now - no_person_started_at >= args.no_person_alert_seconds:
                    no_person_alert_active = True
                    logger.record(
                        "empty_scene",
                        "info",
                        f"No visible person detected for {args.no_person_alert_seconds:.0f}s.",
                        now,
                        metadata={"seconds_without_people": round(now - no_person_started_at, 2)},
                        frame=frame,
                        snapshot_key="empty_scene",
                    )
                    emitted_events += 1
            else:
                if no_person_alert_active:
                    logger.record(
                        "person_presence_resumed",
                        "info",
                        "Person presence resumed after an empty-scene interval.",
                        now,
                        metadata={"visible_people": visible_people},
                        frame=frame,
                        snapshot_key="person_presence_resumed",
                    )
                    emitted_events += 1
                no_person_started_at = None
                no_person_alert_active = False

            if visible_people >= args.crowd_threshold and not crowd_alert_active:
                crowd_alert_active = True
                logger.record(
                    "crowd_detected",
                    "warn",
                    f"Crowd threshold reached with {visible_people} visible people.",
                    now,
                    metadata={"visible_people": visible_people, "threshold": args.crowd_threshold},
                    frame=frame,
                    snapshot_key="crowd_detected",
                )
                emitted_events += 1
            elif visible_people < args.crowd_threshold:
                crowd_alert_active = False

            if any(d["class_name"] in PHONE_CLASS_NAMES for d in other_detections):
                if now - last_phone_alert_at >= args.snapshot_cooldown:
                    last_phone_alert_at = now
                    logger.record(
                        "phone_detected",
                        "info",
                        "A mobile phone was detected in view.",
                        now,
                        metadata={"visible_people": visible_people},
                        frame=frame,
                        snapshot_key="phone_detected",
                    )
                    emitted_events += 1

            emotion_summaries: list[dict[str, Any]] = []
            if emotion_detector is not None:
                emotion_started = time.perf_counter()
                emotion_detections = emotion_detector.detect_emotions(rgb_frame)
                emotion_summaries = draw_emotion_annotations(frame, emotion_detections, tracks)
                timings_ms["emotion"] = (time.perf_counter() - emotion_started) * 1000.0

                strongest_distress = max(
                    (
                        summary
                        for summary in emotion_summaries
                        if summary["emotion"] in DISTRESS_EMOTIONS and summary["score"] >= args.distress_threshold
                    ),
                    key=lambda summary: summary["score"],
                    default=None,
                )
                if strongest_distress is not None and now - last_distress_alert_at >= args.snapshot_cooldown:
                    last_distress_alert_at = now
                    track_suffix = (
                        f" for person #{strongest_distress['track_id']}"
                        if strongest_distress["track_id"] is not None
                        else ""
                    )
                    logger.record(
                        "distress_emotion",
                        "warn",
                        f"Detected {strongest_distress['emotion']} with score {strongest_distress['score']:.2f}{track_suffix}.",
                        now,
                        metadata=strongest_distress,
                        frame=frame,
                        snapshot_key="distress_emotion",
                    )
                    emitted_events += 1

            if pose_bundle is not None:
                pose_started = time.perf_counter()
                pose_result = build_pose_result(pose_bundle, rgb_frame)
                draw_pose(frame, pose_bundle, pose_result)
                timings_ms["pose"] = (time.perf_counter() - pose_started) * 1000.0
                if not args.no_fall:
                    fall_events = detect_fall_events(
                        pose_bundle,
                        pose_result,
                        tracks,
                        frame.shape,
                        now,
                        angle_threshold=args.fall_angle_threshold,
                        aspect_threshold=args.fall_aspect_threshold,
                        cooldown_seconds=args.snapshot_cooldown,
                    )
                    emitted_events += emit_runtime_events(logger, fall_events, now, frame)

            if captioner is not None:
                if frame_index % max(1, args.caption_interval) == 0:
                    captioner.submit(frame_index, rgb_frame)
                latest_caption, latest_caption_ms, latest_caption_error = captioner.latest()
                if latest_caption_error:
                    latest_caption = f"Caption error: {latest_caption_error}"
                timings_ms["caption"] = latest_caption_ms
            else:
                latest_caption = "Scene captioning disabled (--no-caption)."

            draw_other_detections(frame, other_detections)
            draw_person_tracks(frame, tracks, now)
            if not args.no_polygon_zone and polygon_points_px:
                draw_restricted_polygon(frame, polygon_points_px)
            if not args.no_line_crossing and line_value_px is not None:
                draw_intrusion_line(frame, line_value_px, args.line_direction, args.line_orientation)

            fps, previous_time = compute_fps(previous_time)
            average_fps_accumulator += fps
            active_alerts = emitted_events + int(no_person_alert_active) + int(crowd_alert_active)

            draw_top_hud(frame, fps, timings_ms, visible_people, active_alerts)
            draw_event_feed(frame, logger.recent_events)
            draw_caption_bar(frame, latest_caption)

            if writer is not None:
                writer.write(frame)

            if show_window:
                cv2.imshow("AI Smart Surveillance System", frame)
            frame_index += 1

            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if pose_bundle is not None:
            pose_bundle.pose.close()
        if captioner is not None:
            captioner.close()
        if show_window:
            cv2.destroyAllWindows()

    runtime_seconds = max(0.001, time.perf_counter() - session_started_perf)
    average_fps = average_fps_accumulator / max(1, frame_index)
    latest_caption_text = latest_caption if "latest_caption" in locals() else ""
    logger.record(
        "session_ended",
        "info",
        "Surveillance session ended.",
        time.time(),
        metadata={"frames_processed": frame_index, "average_fps": round(average_fps, 2)},
    )
    logger.write_summary(
        runtime_seconds=runtime_seconds,
        frames_processed=frame_index,
        average_fps=average_fps,
        max_people=visible_people_max,
        latest_caption=latest_caption_text,
        output_video_path=str(args.output.resolve()) if args.save else None,
    )

    if args.save:
        print(f"Saved output video to: {args.output.resolve()}")
    print(f"Event log: {session_paths.event_log_path}")
    print(f"Session summary: {session_paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
