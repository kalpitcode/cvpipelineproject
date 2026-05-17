from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "surveillance_runs"


def build_command(
    input_path: Path,
    output_path: Path,
    session_name: str,
    options: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "ai_pipeline.py"),
        "--input-video",
        str(input_path),
        "--no-display",
        "--save",
        "--output",
        str(output_path),
        "--session-name",
        session_name,
    ]

    value_flags = {
        "--yolo-conf": options["yolo_conf"],
        "--caption-interval": options["caption_interval"],
        "--track-max-missed": options["track_max_missed"],
        "--track-distance": options["track_distance"],
        "--loiter-seconds": options["loiter_seconds"],
        "--no-person-alert-seconds": options["no_person_alert_seconds"],
        "--crowd-threshold": options["crowd_threshold"],
        "--distress-threshold": options["distress_threshold"],
        "--snapshot-cooldown": options["snapshot_cooldown"],
        "--line-orientation": options["line_orientation"],
        "--line-direction": options["line_direction"],
        "--intrusion-line-x": options["intrusion_line_x"],
        "--intrusion-line-y": options["intrusion_line_y"],
        "--fall-angle-threshold": options["fall_angle_threshold"],
        "--fall-aspect-threshold": options["fall_aspect_threshold"],
        "--restricted-polygon": options["restricted_polygon"],
    }
    for flag, value in value_flags.items():
        if value is None:
            continue
        if flag == "--restricted-polygon" and not str(value).strip():
            continue
        command.extend([flag, str(value)])

    bool_flags = {
        "--no-emotion": options["no_emotion"],
        "--no-pose": options["no_pose"],
        "--no-caption": options["no_caption"],
        "--no-line-crossing": options["no_line_crossing"],
        "--no-polygon-zone": options["no_polygon_zone"],
        "--no-fall": options["no_fall"],
    }
    for flag, enabled in bool_flags.items():
        if enabled:
            command.append(flag)
    return command


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def main() -> None:
    st.set_page_config(page_title="AI Smart Surveillance Dashboard", layout="wide")
    st.title("AI Smart Surveillance Dashboard")
    st.caption("Upload a video, run the CPU-only surveillance pipeline, and review alerts, evidence, and summary analytics.")

    with st.sidebar:
        st.header("Pipeline Settings")
        yolo_conf = st.slider("YOLO confidence", 0.10, 0.90, 0.25, 0.05)
        caption_interval = st.number_input("Caption interval (frames)", min_value=5, max_value=120, value=30, step=5)
        track_max_missed = st.number_input("Track max missed frames", min_value=1, max_value=120, value=20, step=1)
        track_distance = st.number_input("Track distance", min_value=20, max_value=500, value=140, step=5)
        loiter_seconds = st.number_input("Loiter seconds", min_value=1.0, max_value=120.0, value=10.0, step=1.0)
        no_person_alert_seconds = st.number_input("Empty-scene alert seconds", min_value=1.0, max_value=120.0, value=15.0, step=1.0)
        crowd_threshold = st.number_input("Crowd threshold", min_value=1, max_value=20, value=3, step=1)
        distress_threshold = st.slider("Distress threshold", 0.30, 0.99, 0.72, 0.01)
        snapshot_cooldown = st.number_input("Snapshot cooldown", min_value=1.0, max_value=60.0, value=8.0, step=1.0)

        st.header("Boundary Rules")
        line_orientation = st.selectbox("Line orientation", ["horizontal", "vertical"], index=0)
        if line_orientation == "horizontal":
            line_direction = st.selectbox("Line direction", ["down", "up", "any"], index=0)
        else:
            line_direction = st.selectbox("Line direction", ["right", "left", "any"], index=0)
        intrusion_line_y = st.slider("Horizontal line Y", 0.0, 1.0, 0.55, 0.01)
        intrusion_line_x = st.slider("Vertical line X", 0.0, 1.0, 0.50, 0.01)
        restricted_polygon = st.text_input(
            "Restricted polygon",
            value="",
            help="Normalized points like 0.1,0.2;0.8,0.2;0.8,0.7;0.1,0.7",
        )

        st.header("Advanced Modules")
        fall_angle_threshold = st.slider("Fall angle threshold", 5.0, 60.0, 35.0, 1.0)
        fall_aspect_threshold = st.slider("Fall aspect ratio threshold", 0.5, 2.0, 1.1, 0.05)
        no_emotion = st.checkbox("Disable emotion module", value=False)
        no_pose = st.checkbox("Disable pose module", value=False)
        no_caption = st.checkbox("Disable caption module", value=False)
        no_line_crossing = st.checkbox("Disable line crossing", value=False)
        no_polygon_zone = st.checkbox("Disable polygon zone", value=False)
        no_fall = st.checkbox("Disable fall detection", value=False)

    uploaded_file = st.file_uploader("Upload a video for surveillance analysis", type=["mp4", "mov", "avi", "mkv"])
    process_clicked = st.button("Process Video", type="primary", disabled=uploaded_file is None)

    if not process_clicked:
        st.info("Upload a video and click Process Video to generate annotated output, alerts, and a session summary.")
        return

    if uploaded_file is None:
        st.error("Please upload a video file first.")
        return

    session_name = f"dashboard_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_dir = RUNS_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    input_path = session_dir / uploaded_file.name
    output_path = session_dir / "annotated_output.mp4"
    input_path.write_bytes(uploaded_file.getbuffer())

    options = {
        "yolo_conf": yolo_conf,
        "caption_interval": caption_interval,
        "track_max_missed": track_max_missed,
        "track_distance": track_distance,
        "loiter_seconds": loiter_seconds,
        "no_person_alert_seconds": no_person_alert_seconds,
        "crowd_threshold": crowd_threshold,
        "distress_threshold": distress_threshold,
        "snapshot_cooldown": snapshot_cooldown,
        "line_orientation": line_orientation,
        "line_direction": line_direction,
        "intrusion_line_x": intrusion_line_x,
        "intrusion_line_y": intrusion_line_y,
        "restricted_polygon": restricted_polygon,
        "fall_angle_threshold": fall_angle_threshold,
        "fall_aspect_threshold": fall_aspect_threshold,
        "no_emotion": no_emotion,
        "no_pose": no_pose,
        "no_caption": no_caption,
        "no_line_crossing": no_line_crossing,
        "no_polygon_zone": no_polygon_zone,
        "no_fall": no_fall,
    }
    command = build_command(input_path, output_path, session_name, options)

    with st.spinner("Running surveillance analysis... this can take a while on CPU."):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    if completed.returncode != 0:
        st.error("Pipeline execution failed.")
        if completed.stdout:
            st.code(completed.stdout, language="text")
        if completed.stderr:
            st.code(completed.stderr, language="text")
        return

    summary_path = session_dir / "session_summary.json"
    event_log_path = session_dir / "events.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    events = load_jsonl(event_log_path)

    st.success("Processing finished.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames processed", summary.get("frames_processed", 0))
    col2.metric("Average FPS", summary.get("average_fps", 0))
    col3.metric("Max people observed", summary.get("max_people_observed", 0))

    st.subheader("Annotated Output")
    if output_path.exists():
        st.video(str(output_path))
        st.download_button("Download annotated video", data=output_path.read_bytes(), file_name=output_path.name, mime="video/mp4")
    else:
        st.warning("Annotated video was not generated.")

    st.subheader("Session Summary")
    st.json(summary)

    st.subheader("Event Log")
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("No events were logged for this run.")

    snapshot_dir = session_dir / "snapshots"
    snapshot_paths = sorted(snapshot_dir.glob("*.jpg"))
    st.subheader("Evidence Snapshots")
    if snapshot_paths:
        for snapshot_path in snapshot_paths:
            st.image(str(snapshot_path), caption=snapshot_path.name, use_container_width=True)
    else:
        st.info("No snapshots were captured during this run.")

    with st.expander("Processing Console Output"):
        if completed.stdout:
            st.code(completed.stdout, language="text")
        if completed.stderr:
            st.code(completed.stderr, language="text")


if __name__ == "__main__":
    main()
