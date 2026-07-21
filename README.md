# Integrated Live Demo
<img width="1069" height="598" alt="image" src="https://github.com/user-attachments/assets/86a42448-1c48-4964-a3bd-06a802892731" />

Live webcam demo combining:

- multi-person YOLO pose tracking and skeleton rendering
- repeated raised right-hand wave activity tiers
- MediaPipe face detection
- EmotiEffLib facial-expression estimates
- YOLO person tracking
- optional enrolled-person identification with OpenCV YuNet + SFace

Each tracked person receives an independent skeleton, wave counter, expression
context, review overlay, and optional identity label. Model downloads are
atomic and checksum-verified before use.

## Identity Matching
<img width="1118" height="624" alt="image" src="https://github.com/user-attachments/assets/546550a0-efeb-4ee6-baf0-524638fbcb93" />

Identity matching no longer depends on `dlib` or the `face-recognition`
package. The default stack uses OpenCV's DNN APIs:

- `FaceDetectorYN` with `data/face_detection_yunet_2023mar.onnx`
- `FaceRecognizerSF` with `data/face_recognition_sface_2021dec.onnx`

This is easier to install on Windows and is also portable to Linux and macOS
through `opencv-contrib-python`.
These two OpenCV model files are downloaded automatically on first use if they
are not already present. Identity is confirmed only after multiple agreeing
frames, poor-quality faces are rejected, and confirmed names expire unless
they are revalidated.

The EmotiEffLib expression model is also downloaded from a pinned upstream
revision and checksum-verified before it is loaded. Once the models have been
prepared, `--no-model-downloads` provides a fully offline startup check.

If no enrollments are available, the live monitoring demo continues
anonymously.

## Review Indicator
<img width="545" height="860" alt="image" src="https://github.com/user-attachments/assets/f0dc70a0-f4f5-42e6-9180-1b26db5841f3" />

The overlay begins green. Completed right-hand wave gestures are counted in a
rolling 30-second window, with two waves ignored as likely ordinary activity.
Review color uses a fixed professional ladder:

- `CLEAR` - green
- `MONITOR` - yellow
- `REVIEW` - orange
- `HIGH` - red

The best expression estimate remains visible even at lower confidence; a `?`
marks a tentative label. Per-person estimates use a short rolling vote to
reduce flicker. High-confidence `Anger`, `Contempt`, `Disgust`, `Fear`, and
`Sadness` estimates also appear as a separate expression-context signal.
Expression does not change the behavior-based review tier, and both the label
and context clear when the face is no longer visible.

This is a demo review indicator, not a conclusion that a person is suspicious
or dangerous. Facial-expression estimates are uncertain and can reflect many
ordinary situations.

## Run

Python 3.11 or 3.12 is recommended. Create an environment and install the
tested dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe src\main.py --doctor --prepare-models
```

On macOS or Linux, use `.venv/bin/python` instead of
`.venv\Scripts\python.exe`.

```powershell
python src/main.py
```

Use `enroll` to add identity photos and `detect` to start live monitoring.
Press `q` to close the camera window.

The live HUD shows FPS, active people, identity availability, and recent
operator events. Detailed logs are written to `output/logs`, and meaningful
tier, identity, wave, and expression changes are written as JSONL files under
`output/events`.

To bypass the menu and launch a camera directly:

```powershell
python src/main.py --detect --source 1
```

Camera source `0` is usually the built-in webcam. Source `1` is often an
external USB camera.

## Export An Annotated Photo Or Video

Place a photo or recorded video in the `input` folder, then run:

```powershell
python src/main.py --process-media
```

Annotated copies are written to the `output` folder. The command supports
photos (`.bmp`, `.jpeg`, `.jpg`, `.png`, `.webp`) and videos (`.avi`, `.m4v`,
`.mkv`, `.mov`, `.mp4`).

On Windows, you can also double-click `Run Media Export.cmd`.

You can also process one specific file:

```powershell
python src/main.py --process-media --input "C:\path\to\photo.jpg"
```

## Demo-Only High Review Override

For controlled demos, an enrolled identity can be displayed in the red `HIGH`
tier without waiting for repeated activity to accumulate:

```powershell
$env:DEMO_HIGH_REVIEW_NAMES = "Taylor Brooks"
python src/main.py --process-media --input "input\demo.png"
```

This override is only for staged demonstrations. A visible `DEMO OVERRIDE
ACTIVE` watermark is drawn whenever it affects a person. Leave
`DEMO_HIGH_REVIEW_NAMES` unset for normal runs.

## Validate The POC

Copy `validation/manifest.example.json` to `validation/manifest.json`, add
consented recorded scenarios under `validation/media`, then run:

```powershell
.\.venv\Scripts\python.exe src\main.py --validate `
  --manifest validation\manifest.json `
  --report validation\reports\latest.json
```

The runner measures people detected, confirmed identities, tier counts, and
processing FPS, and returns a failing exit code when a scenario misses its
acceptance criteria. See `validation/README.md` for the recommended scenario
set.

## Runtime Configuration

Common environment overrides include:

- `MONITOR_ALLOW_MODEL_DOWNLOADS=false` for verified offline operation
- `MONITOR_IDENTITY_THRESHOLD=0.363` for a calibrated SFace threshold
- `MONITOR_EXPRESSION_CONFIDENCE=0.65` for expression context visibility
- `MONITOR_EXPRESSION_DISPLAY_CONFIDENCE=0.45` for tentative `?` labels
- `MONITOR_EXPRESSION_SMOOTHING_SECONDS=0.8` for display stability
- `MONITOR_EVENT_LOGGING=false` to disable JSONL event reports
- `MONITOR_DEBUG_TIMING=true` to log stage timings

Run `python src/main.py --doctor` at any time to check the local installation.
