"""
Event batching and rate limiting for SSE streaming.

Provides utilities to batch rapid events (especially progress updates)
to avoid overwhelming the frontend with too many updates.
"""

from __future__ import annotations

import time
from collections import deque

from backend.api.streaming.events import SSEEvent, SSEEventType


class EventBatcher:
    """
    Batch rapid events to avoid overwhelming the frontend.

    Combines multiple TOOL_PROGRESS events within a time window.
    Non-progress events are emitted immediately.
    """

    def __init__(self, batch_window_ms: int = 100) -> None:
        """
        Initialize the event batcher.

        Args:
            batch_window_ms: Time window in milliseconds for batching
                progress events. Events within this window are combined.
        """
        self.batch_window = batch_window_ms / 1000  # Convert to seconds
        self._pending: deque[SSEEvent] = deque()
        self._last_emit: float = 0

    def add(self, event: SSEEvent) -> SSEEvent | None:
        """
        Add event, return it if should emit immediately.

        Progress events are batched; other events emit immediately.

        Args:
            event: The SSE event to add.

        Returns:
            The event if it should be emitted immediately, None if batched.
        """
        now = time.time()

        # Non-progress events emit immediately
        if event.type != SSEEventType.TOOL_PROGRESS:
            return event

        # Check if we should emit batched progress
        if now - self._last_emit >= self.batch_window:
            self._last_emit = now
            # Return latest progress (discard intermediate)
            self._pending.clear()
            return event

        # Queue for later
        self._pending.append(event)
        return None

    def flush(self) -> list[SSEEvent]:
        """
        Get all pending events and clear the queue.

        Returns:
            List of pending events.
        """
        events = list(self._pending)
        self._pending.clear()
        return events

    def get_latest_progress(self) -> SSEEvent | None:
        """
        Get the most recent progress event without clearing.

        Useful for checking current progress state.

        Returns:
            Latest progress event or None if empty.
        """
        if self._pending:
            return self._pending[-1]
        return None

    @property
    def pending_count(self) -> int:
        """Number of pending events in the queue."""
        return len(self._pending)

    def clear(self) -> None:
        """Clear all pending events."""
        self._pending.clear()
        self._last_emit = 0


class RateLimiter:
    """
    Rate limiter for SSE events by event type.

    Ensures events of a specific type are not sent more frequently
    than the specified interval.
    """

    def __init__(self, default_interval_ms: int = 100) -> None:
        """
        Initialize the rate limiter.

        Args:
            default_interval_ms: Default minimum interval between events
                of the same type, in milliseconds.
        """
        self._default_interval = default_interval_ms / 1000
        self._last_emit: dict[SSEEventType, float] = {}
        self._intervals: dict[SSEEventType, float] = {}

    def set_interval(self, event_type: SSEEventType, interval_ms: int) -> None:
        """
        Set custom interval for a specific event type.

        Args:
            event_type: The event type to configure.
            interval_ms: Minimum interval in milliseconds.
        """
        self._intervals[event_type] = interval_ms / 1000

    def should_emit(self, event: SSEEvent) -> bool:
        """
        Check if an event should be emitted based on rate limits.

        Args:
            event: The SSE event to check.

        Returns:
            True if the event should be emitted.
        """
        now = time.time()
        event_type = event.type

        # Get interval for this event type
        interval = self._intervals.get(event_type, self._default_interval)

        # Check last emit time
        last_emit = self._last_emit.get(event_type, 0)

        if now - last_emit >= interval:
            self._last_emit[event_type] = now
            return True

        return False

    def reset(self, event_type: SSEEventType | None = None) -> None:
        """
        Reset rate limit tracking.

        Args:
            event_type: If provided, reset only this type.
                If None, reset all types.
        """
        if event_type is None:
            self._last_emit.clear()
        elif event_type in self._last_emit:
            del self._last_emit[event_type]


# Default configuration for event rate limiting
DEFAULT_RATE_LIMITS = {
    SSEEventType.TOOL_PROGRESS: 100,  # 100ms between progress updates
    SSEEventType.HEARTBEAT: 30000,  # 30s between heartbeats
    SSEEventType.THINKING: 500,  # 500ms between thinking updates
}


def create_default_rate_limiter() -> RateLimiter:
    """
    Create a rate limiter with default configuration.

    Returns:
        Configured RateLimiter instance.
    """
    limiter = RateLimiter()
    for event_type, interval in DEFAULT_RATE_LIMITS.items():
        limiter.set_interval(event_type, interval)
    return limiter
