# Integrated Live Demo

Live webcam demo combining:

- MoveNet Lightning skeleton rendering
- repeated raised right-hand wave activity tiers
- MediaPipe face detection
- EmotiEffLib facial-expression estimates
- YOLO person tracking with optional enrolled-person identification

YOLO can display multiple detected people. The pose, wave, expression
modifier, and review overlay currently apply to the single person selected by
MoveNet SinglePose.

## Review indicator

The overlay begins green. Completed right-hand wave gestures are counted in a
rolling 30-second window, with two waves ignored as likely ordinary activity.
Review color uses a fixed professional ladder:

- `CLEAR` — green
- `MONITOR` — yellow
- `REVIEW` — orange
- `HIGH` — red

High-confidence `Anger`, `Contempt`, `Disgust`, `Fear`, and `Sadness`
estimates increase a visible, smoothed expression modifier from `x1.00`
toward `x1.50`. The modifier strengthens repeated-wave activity only; an
expression estimate without repeated activity does not raise the review level.
It clears when the face is no longer visible.

This is a demo review indicator, not a conclusion that a person is suspicious
or dangerous. Facial-expression estimates are uncertain and can reflect many
ordinary situations.

## Expression labels

The expression model reports one of: `Anger`, `Contempt`, `Disgust`, `Fear`,
`Happiness`, `Neutral`, `Sadness`, or `Surprise`.

Identity enrollment is optional. If enrolled identities are unavailable, the
live monitoring demo continues anonymously.

## Run

```powershell
python src/main.py
```

Press `q` to close the camera window.

