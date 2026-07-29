# Integrated Live Demo
<img width="1069" height="598" alt="image" src="https://github.com/user-attachments/assets/86a42448-1c48-4964-a3bd-06a802892731" />

Live webcam demo combining:

- multi-person YOLO pose tracking and skeleton rendering
- repeated raised right-hand wave activity tiers
- MediaPipe face detection
- EmotiEffLib facial-expression estimates
- YOLO person tracking
- optional enrolled-person identification with OpenCV YuNet + SFace
- optional per-person MLP activity labels from existing YOLO Pose keypoints

Each tracked person receives an independent skeleton, wave counter, expression
context, review overlay, and optional identity and activity labels. Activity
recognition is disabled by default and remains informational: its output is not
an input to review-tier scoring. Model downloads are atomic and
checksum-verified before use.

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

### Optional Live Activity Labels

Live MLP activity recognition supports `walking`, `running`, `standing`, and
`sitting`. It reuses the YOLO Pose results already produced for each ByteTrack
ID; it does not run a second pose model. Install the optional dependencies and
provide a compatible checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-activity.txt
.\.venv\Scripts\python.exe src\main.py --detect --source 0 `
  --activity-model mlp `
  --activity-checkpoint "data\activity_models\mlp.pt"
```

The default checkpoint path is `data/activity_models/mlp.pt`, which is created
by `scripts/train_activity_models.py` and intentionally excluded from version
control. The checkpoint must contain the canonical label order and the same
16-frame sequence length used by the live configuration.

The live runtime waits for 16 valid tracked poses, performs MLP inference every
five frames by default, and averages the five most recent probability vectors.
Predictions below the default `0.50` confidence threshold are displayed as
`Unknown`. Missing poses do not invoke the classifier, and an all-zero history
is handled as `Unknown`. History and smoothing state are independent per track
and are removed when a track expires or tracking resets.

Activity labels and confidence appear in the per-person overlay and optional
event report. They do not increase or decrease review tiers.

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
python src/main.py --process-media --input "input\photo.jpg"
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
- `MONITOR_ACTIVITY_MODEL=none` or `mlp`
- `MONITOR_ACTIVITY_CHECKPOINT=data/activity_models/mlp.pt`
- `MONITOR_ACTIVITY_SEQUENCE_LENGTH=16`
- `MONITOR_ACTIVITY_CONFIDENCE=0.50`
- `MONITOR_ACTIVITY_INTERVAL=5`
- `MONITOR_ACTIVITY_SMOOTHING_WINDOW=5`
- `MONITOR_EVENT_LOGGING=false` to disable JSONL event reports
- `MONITOR_DEBUG_TIMING=true` to log stage timings

Run `python src/main.py --doctor` at any time to check the local installation.

## Human Activity Recognition Benchmark

The repository includes an optional benchmark for four human activities:
`walking`, `running`, `standing`, and `sitting`. The MLP checkpoint can be
enabled in the live monitoring pipeline as described above. MobileNetV2 and
S3D remain offline comparison models. Existing behavior is unchanged when the
activity model is `none`.

Three compact approaches use the same source-video partitions:

- **MLP:** a small network trained on 16 frames of bounding-box-normalized YOLO
  Pose coordinates and visibility values.
- **CNN:** a four-class linear head trained on cached features from a frozen
  ImageNet-pretrained MobileNetV2.
- **Advanced video model:** a four-class linear head trained on cached features
  from a frozen Kinetics-400-pretrained S3D model.

The frozen feature caches make repeated head training fast and avoid rerunning
YOLO or the pretrained backbones every epoch. TensorFlow is not used by this
benchmark.

### Dataset

The benchmark expects the official
[HMDB51](https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)
videos and
[official split files](https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar),
restricted to the `walk`, `run`, `stand`, and `sit` classes. Fold 1 is the
default. Official test clips remain untouched; the official training clips are
divided deterministically into training and validation sets at the video level.
Derived frames from one source video never cross partitions.

HMDB51 contains clips collected from varied movie and internet-video sources.
Its distribution does not present a simple permissive software-style license.
Review the source terms before commercial use, do not redistribute the videos,
and keep the dataset outside version control.

### Model And Dataset Sources

- Pose extraction uses
  [Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/models/yolov8/).
  Ultralytics documents AGPL-3.0 and enterprise licensing options for its code
  and pretrained models; verify that the selected option fits the deployment.
- The image backbone uses Torchvision
  [MobileNetV2](https://docs.pytorch.org/vision/stable/models/mobilenetv2.html)
  with `MobileNet_V2_Weights.DEFAULT`.
- The video backbone uses Torchvision
  [S3D](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.s3d.html)
  with `S3D_Weights.KINETICS400_V1`. Torchvision marks its video-model API as
  beta.
- Torchvision source code uses the
  [BSD 3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).
  Its maintainers note that pretrained weights can also carry terms inherited
  from their training data. Review model and dataset terms for the intended
  use.

Extract the video archive and split archive, then prepare the one-time cache:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-activity.txt
.\.venv\Scripts\python.exe scripts\prepare_activity_data.py `
  --dataset-root "data\hmdb51\hmdb51_org" `
  --annotations-root "data\hmdb51\splits"
```

When the original archive endpoint is unavailable, the fold-1 CSV metadata and
individual clips from the
[HMDB51 Hugging Face mirror](https://huggingface.co/datasets/Sina272/hmdb51-v2)
can be used to download only the four target classes:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_activity_data.py `
  --dataset-root "data\hmdb51\subset" `
  --metadata-train "data\hmdb51\metadata_train.csv" `
  --metadata-test "data\hmdb51\metadata_test.csv" `
  --download-videos
```

Train all three heads and immediately evaluate the official test split:

```powershell
.\.venv\Scripts\python.exe scripts\train_activity_models.py
```

Evaluate existing checkpoints without retraining:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_activity_models.py
```

Measured outputs are written to `output/activity_results`:

- `metrics.json`
- `model_comparison.csv`
- `predictions.csv`
- `mlp_confusion_matrix.png`
- `cnn_confusion_matrix.png`
- `advanced_confusion_matrix.png`
- `training_curves.png`
- `run_manifest.json`

Accuracy, macro precision, macro recall, macro F1, confusion matrices, head
training time, and measured inference latency are calculated from real
predictions. Dataset files, caches, checkpoints, and generated reports are
ignored by Git.

The measured experiment and its interpretation are recorded in
[`docs/activity_recognition_results.md`](docs/activity_recognition_results.md).

Benchmark training and evaluation remain offline. Only the saved MLP head is
available for optional live inference.
