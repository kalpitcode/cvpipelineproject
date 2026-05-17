# AI Smart Surveillance System

Resume-ready real-time computer vision surveillance system built in Python with:

- YOLOv8n for object and person detection
- FER for facial emotion recognition
- MediaPipe Pose for pose estimation
- BLIP for scene captioning
- Streamlit for video-upload analysis and review

This project goes beyond a basic webcam demo. It combines multi-model inference, person tracking, rule-based alerting, evidence capture, structured event logging, and a web dashboard for reviewing uploaded videos.

## Why This Is Resume-Worthy

This project demonstrates:

- real-time multi-model inference on CPU
- person tracking with persistent IDs
- event-driven surveillance logic
- uploaded-video analysis workflow
- asynchronous caption generation
- evidence snapshot capture
- structured JSON logging and session summaries
- local dashboard deployment with Streamlit

## Core Features

- Live webcam surveillance mode
- Uploaded-video analysis mode
- Streamlit dashboard for user-uploaded videos
- Person tracking with persistent track IDs
- Loitering detection
- Empty-scene and crowd alerts
- Distress-emotion alerts
- Phone detection alerts
- Horizontal or vertical line-crossing intrusion detection
- Polygon-based restricted zone entry detection
- Pose-based fall detection
- Background BLIP scene captioning
- Evidence snapshots for important alerts
- JSONL event logs and JSON session summaries
- Optional annotated video export
- CPU-only execution

## Example Events

- `person_entered`
- `person_left`
- `loitering_detected`
- `empty_scene`
- `person_presence_resumed`
- `crowd_detected`
- `phone_detected`
- `distress_emotion`
- `line_crossing_intrusion`
- `restricted_zone_entry`
- `fall_detected`

## Project Structure

- `ai_pipeline.py`: main surveillance pipeline CLI
- `streamlit_app.py`: Streamlit dashboard for uploaded-video analysis
- `requirements.txt`: install entrypoint
- `requirements-py311.txt`: pinned dependency set for Python 3.11
- `setup_env.ps1`: Windows environment bootstrap script
- `SETUP.md`: setup notes
- `Dockerfile`: Streamlit deployment target
- `surveillance_runs/`: generated event logs, summaries, snapshots, and outputs

## Modes

### 1. Live Webcam Surveillance

Use your webcam as the input source and monitor the feed in real time.

```powershell
.\.
venv311\Scripts\Activate.ps1
python ai_pipeline.py
```

### 2. Uploaded Video Analysis from CLI

Analyze a local video file without opening the live preview window.

```powershell
python ai_pipeline.py --input-video "sample.mp4" --no-display --save --output "annotated_sample.mp4"
```

This is the easiest mode to use for portfolio demos, testing, and deployment.

### 3. Streamlit Dashboard

Run a local dashboard where a user uploads a video, the system processes it, and the app returns:

- annotated output video
- session summary
- alert/event table
- evidence snapshots

```powershell
streamlit run streamlit_app.py
```

## Boundary and Zone Examples

Horizontal line:

```powershell
python ai_pipeline.py --line-orientation horizontal --intrusion-line-y 0.6 --line-direction down
```

Vertical line:

```powershell
python ai_pipeline.py --line-orientation vertical --intrusion-line-x 0.45 --line-direction right
```

Polygon restricted zone:

```powershell
python ai_pipeline.py --restricted-polygon "0.1,0.2;0.8,0.2;0.8,0.7;0.1,0.7"
```

Combined example:

```powershell
python ai_pipeline.py --input-video "sample.mp4" --no-display --save --output "annotated_sample.mp4" --line-orientation vertical --intrusion-line-x 0.45 --line-direction right --restricted-polygon "0.1,0.2;0.8,0.2;0.8,0.7;0.1,0.7"
```

## Useful CLI Options

```text
--input-video             Analyze a video file instead of a webcam
--no-display              Run headless without OpenCV preview
--save                    Save annotated output video
--output                  Output path for the saved video
--session-name            Custom run folder name
--log-dir                 Root folder for logs and evidence

--line-orientation        horizontal or vertical tripwire
--intrusion-line-y        horizontal line position
--intrusion-line-x        vertical line position
--line-direction          crossing direction to monitor
--restricted-polygon      restricted zone polygon in normalized coordinates
--no-line-crossing        disable tripwire intrusion alerts
--no-polygon-zone         disable polygon zone alerts

--fall-angle-threshold    pose angle threshold for fall detection
--fall-aspect-threshold   aspect-ratio threshold for fall detection
--no-fall                 disable fall detection

--no-emotion              disable FER
--no-pose                 disable MediaPipe Pose
--no-caption              disable BLIP
```

## Session Artifacts

Each run creates a folder in `surveillance_runs/` containing:

- `events.jsonl`: event stream
- `session_summary.json`: aggregate run metrics
- `snapshots/`: evidence images for alerts
- annotated output video when `--save` is enabled

## Local Setup

### Recommended

Use the prepared Python 3.11 environment.

### Rebuild from scratch

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1 -ForceRecreate
```

### Manual install

```powershell
py -3.11 -m venv .venv311
.\.
venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
python -m pip install -r requirements.txt
```

## Streamlit Dashboard Usage

Run:

```powershell
streamlit run streamlit_app.py
```

Then:

1. upload a video
2. choose thresholds and rule settings in the sidebar
3. click `Process Video`
4. review the annotated video, event log, summary, and evidence snapshots

## Docker Deployment

This repo includes a Dockerfile for the Streamlit dashboard deployment path.

Build:

```bash
docker build -t ai-smart-surveillance .
```

Run:

```bash
docker run -p 8501:8501 ai-smart-surveillance
```

Then open:

```text
http://localhost:8501
```

## Deployment Recommendation

For this project, the best deployment path is:

- webcam mode for local demos
- uploaded-video mode for user-facing web interaction
- Streamlit dashboard for review and presentation

This is more practical than trying to host a heavy live-webcam surveillance pipeline directly in a public browser app.

## Notes

- First run may download model weights and the MediaPipe pose model file.
- BLIP captioning runs asynchronously to keep the surveillance loop responsive.
- `fer` currently depends on `pkg_resources`, so this project pins `setuptools<81`.
- Press `q` or `Esc` to quit the OpenCV preview window in live mode.
