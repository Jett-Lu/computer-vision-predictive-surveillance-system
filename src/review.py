from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gesture import GREEN, WaveAlertState


CONCERN_EXPRESSIONS = frozenset(
    {
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Sadness",
    }
)


@dataclass(frozen=True)
class ReviewState:
    tier_label: str
    color: tuple[int, int, int]
    score: int
    recent_wave_count: int
    recent_expression_event_count: int
    expression_points: int


@dataclass
class ReviewLevelMonitor:
    """Combine repeated behavior and sustained expression cues for on-screen review."""

    expression_window_seconds: float = 30.0
    required_concern_seconds: float = 1.5
    expression_event_cooldown_seconds: float = 4.0
    confidence_threshold: float = 0.55
    expression_allowance: int = 1
    max_expression_points: int = 2
    high_score: int = 5
    _expression_events: deque[float] = field(default_factory=deque, init=False)
    _concern_start_time: float | None = field(default=None, init=False)
    _last_expression_event_time: float | None = field(default=None, init=False)

    def observe_expression(
        self,
        label: str | None,
        confidence: float | None,
        timestamp: float,
    ) -> bool:
        """Record qualifying sustained expression cues; return true when one is counted."""
        self._discard_expired_expression_events(timestamp)

        concerning = (
            label in CONCERN_EXPRESSIONS
            and confidence is not None
            and confidence >= self.confidence_threshold
        )
        if not concerning:
            self._concern_start_time = None
            return False

        if self._concern_start_time is None:
            self._concern_start_time = timestamp
            return False

        duration_met = (
            timestamp - self._concern_start_time >= self.required_concern_seconds
        )
        cooled_down = self._last_expression_event_time is None or (
            timestamp - self._last_expression_event_time
            >= self.expression_event_cooldown_seconds
        )
        if duration_met and cooled_down:
            self._expression_events.append(timestamp)
            self._last_expression_event_time = timestamp
            return True

        return False

    def update(self, wave_state: WaveAlertState, timestamp: float) -> ReviewState:
        self._discard_expired_expression_events(timestamp)

        wave_points = max(0, wave_state.recent_wave_count - 2)
        expression_points = min(
            max(0, len(self._expression_events) - self.expression_allowance),
            self.max_expression_points,
        )
        score = wave_points + expression_points

        if score == 0:
            tier = "CLEAR"
        elif score <= 2:
            tier = "MONITOR"
        elif score <= 4:
            tier = "REVIEW"
        else:
            tier = "HIGH"

        red_progress = min(score / max(1, self.high_score), 1.0)
        red = int(round(255 * red_progress))
        green = int(round(255 * (1.0 - red_progress)))
        color = (0, green, red) if score else GREEN

        return ReviewState(
            tier_label=tier,
            color=color,
            score=score,
            recent_wave_count=wave_state.recent_wave_count,
            recent_expression_event_count=len(self._expression_events),
            expression_points=expression_points,
        )

    def _discard_expired_expression_events(self, timestamp: float) -> None:
        while (
            self._expression_events
            and timestamp - self._expression_events[0] > self.expression_window_seconds
        ):
            self._expression_events.popleft()
