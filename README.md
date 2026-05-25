# Integrated Live Demo

Single-person webcam demo combining:

- MoveNet Lightning skeleton rendering
- repeated raised right-hand wave activity tiers
- MediaPipe face detection
- EmotiEffLib facial-expression estimates

The review color changes from green toward red as repeated right-hand waving
and sustained concern-expression cues are counted in rolling windows.

## Review indicator

The overlay begins green. Completed right-hand wave gestures are counted in a
rolling 30-second window, with two waves ignored as likely ordinary activity:

- three to four recent waves raises the review level to `MONITOR`
- five to six recent waves raises it to `REVIEW`
- seven or more recent waves raises it to `HIGH` (red)

High-confidence `Anger`, `Contempt`, `Disgust`, `Fear`, and `Sadness`
estimates only contribute after they remain visible for at least 1.5 seconds.
One sustained expression cue is ignored, and expression cues alone are capped
at `MONITOR`; they can raise the tier further only when repeated-wave activity
is also present. Older cues expire, so the color can recover toward green.

This is a demo review indicator, not a conclusion that a person is suspicious
or dangerous. Facial-expression estimates are uncertain and can reflect many
ordinary situations.

## Expression labels

The expression model reports one of: `Anger`, `Contempt`, `Disgust`, `Fear`,
`Happiness`, `Neutral`, `Sadness`, or `Surprise`.

This first integrated demo follows one visible person because MoveNet Lightning
is a single-pose model.

## Run

```powershell
python src/main.py
```

Press `q` to close the camera window.

