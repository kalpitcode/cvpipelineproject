# AI Pipeline Setup

This project runs best in a fresh Python 3.11 virtual environment on Windows.

## Why not the current Python 3.12 install?

Your current machine only has Python 3.12.4 installed, and the existing package set includes an old `wrapt` version that breaks the `mediapipe` import path used by pose estimation.

## Recommended Python version

Install **Python 3.11.9 (64-bit)** on Windows.

Official Python release page:

https://www.python.org/downloads/release/python-3119/

Reason:

- It is the last Python 3.11 Windows release with official binary installers.
- It avoids the import issue currently seen in the existing 3.12 environment.

## Automated setup

After installing Python 3.11.9, from this folder run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1
```

If you need to rebuild the environment from scratch later:

```powershell
.\setup_env.ps1 -ForceRecreate
```

## Manual setup

```powershell
py -3.11 -m venv .venv311
.\.
venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
python -m pip install -r requirements-py311.txt
python ai_pipeline.py
```

Note:

- `fer` currently expects `pkg_resources`, so the project pins `setuptools<81`.
- The first live run may also download the MediaPipe pose landmarker model file.

## Optional module toggles

```powershell
python ai_pipeline.py --no-caption
python ai_pipeline.py --no-pose
python ai_pipeline.py --no-emotion
python ai_pipeline.py --save
```

## Dashboard Mode

To launch the uploaded-video dashboard after setup:

```powershell
streamlit run streamlit_app.py
```
