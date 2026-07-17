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
modifier, review overlay, and optional identity label. The
`data/yolov8n-pose.pt` model is downloaded automatically on first use if it is
not already present.

## Identity Matching
<img width="1118" height="624" alt="image" src="https://github.com/user-attachments/assets/546550a0-efeb-4ee6-baf0-524638fbcb93" />

Identity matching no longer depends on `dlib` or the `face-recognition`
package. The default stack uses OpenCV's DNN APIs:

- `FaceDetectorYN` with `data/face_detection_yunet_2023mar.onnx`
- `FaceRecognizerSF` with `data/face_recognition_sface_2021dec.onnx`

This is easier to install on Windows and is also portable to Linux and macOS
through `opencv-contrib-python`.
These two OpenCV model files are downloaded automatically on first use if they
are not already present.

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

High-confidence `Anger`, `Contempt`, `Disgust`, `Fear`, and `Sadness`
estimates increase a visible, smoothed expression modifier from `x1.00`
toward `x1.50`. The modifier strengthens repeated-wave activity only; an
expression estimate without repeated activity does not raise the review level.
It clears when the face is no longer visible.

This is a demo review indicator, not a conclusion that a person is suspicious
or dangerous. Facial-expression estimates are uncertain and can reflect many
ordinary situations.

## Run

```powershell
python src/main.py
```

Use `enroll` to add identity photos and `detect` to start live monitoring.
Press `q` to close the camera window.

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

This override is only for staged demonstrations. Leave
`DEMO_HIGH_REVIEW_NAMES` unset for normal runs.
