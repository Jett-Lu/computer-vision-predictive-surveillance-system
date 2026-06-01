# Integrated Live Demo

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
