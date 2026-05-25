# Body Recognition

Webcam demo for drawing a MoveNet Lightning skeleton over one visible person and
tracking repeated raised right-hand wave activity.

## Repeated-wave activity indicator

The overlay begins green. Completed right-hand wave gestures are counted in a
rolling 30-second window:

- `CLEAR` (green): zero to two recent waves
- `MONITOR`: three to four recent waves
- `REVIEW`: five to six recent waves
- `HIGH` (red): seven or more recent waves

This is an explainable review indicator, not a conclusion that a person is
suspicious. Older gestures expire from the window, so the color can recover
back toward green.

## Run

```powershell
python src/main.py
```

Press `q` to close the camera window.

