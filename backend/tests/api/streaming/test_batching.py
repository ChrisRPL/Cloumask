"""Tests for SSE event batching and rate limiting."""

import time

from backend.api.streaming.batching import (
    DEFAULT_RATE_LIMITS,
    EventBatcher,
    RateLimiter,
    create_default_rate_limiter,
)
from backend.api.streaming.events import (
    SSEEventType,
    message_event,
    tool_progress_event,
)


class TestEventBatcher:
    """Tests for EventBatcher class."""

    def test_batcher_immediate_non_progress(self) -> None:
        """Non-progress events should emit immediately."""
        batcher = EventBatcher()
        event = message_event("assistant", "Hello")

        result = batcher.add(event)
        assert result is event

    def test_batcher_first_progress_emits(self) -> None:
        """First progress event should emit immediately."""
        batcher = EventBatcher(batch_window_ms=100)
        event = tool_progress_event("scan", 0, 1, 100)

        result = batcher.add(event)
        assert result is event

    def test_batcher_batches_rapid_progress(self) -> None:
        """Rapid progress events should be batched."""
        batcher = EventBatcher(batch_window_ms=100)

        # First progress emits
        e1 = tool_progress_event("scan", 0, 1, 100)
        result1 = batcher.add(e1)
        assert result1 is e1

        # Immediate follow-up is batched (within 100ms)
        e2 = tool_progress_event("scan", 0, 2, 100)
        result2 = batcher.add(e2)
        assert result2 is None

        # Check pending count
        assert batcher.pending_count == 1

    def test_batcher_flush_returns_pending(self) -> None:
        """Flush should return all pending events."""
        batcher = EventBatcher(batch_window_ms=100)

        # First emits, rest are batched
        batcher.add(tool_progress_event("scan", 0, 1, 100))
        batcher.add(tool_progress_event("scan", 0, 2, 100))
        batcher.add(tool_progress_event("scan", 0, 3, 100))

        pending = batcher.flush()
        assert len(pending) == 2  # 2 were batched

    def test_batcher_flush_clears_pending(self) -> None:
        """Flush should clear the pending queue."""
        batcher = EventBatcher(batch_window_ms=100)

        batcher.add(tool_progress_event("scan", 0, 1, 100))
        batcher.add(tool_progress_event("scan", 0, 2, 100))

        batcher.flush()
        assert batcher.pending_count == 0

    def test_batcher_emits_after_window(self) -> None:
        """Progress should emit after batch window expires."""
        batcher = EventBatcher(batch_window_ms=10)  # 10ms window

        e1 = tool_progress_event("scan", 0, 1, 100)
        batcher.add(e1)

        # Wait longer than batch window
        time.sleep(0.02)

        e2 = tool_progress_event("scan", 0, 2, 100)
        result = batcher.add(e2)
        assert result is e2  # Should emit after window

    def test_batcher_get_latest_progress(self) -> None:
        """get_latest_progress should return most recent."""
        batcher = EventBatcher(batch_window_ms=100)

        batcher.add(tool_progress_event("scan", 0, 1, 100))
        batcher.add(tool_progress_event("scan", 0, 50, 100))
        batcher.add(tool_progress_event("scan", 0, 75, 100))

        latest = batcher.get_latest_progress()
        assert latest is not None
        assert latest.data["current"] == 75

    def test_batcher_get_latest_progress_empty(self) -> None:
        """get_latest_progress should return None when empty."""
        batcher = EventBatcher()
        assert batcher.get_latest_progress() is None

    def test_batcher_clear(self) -> None:
        """Clear should reset the batcher."""
        batcher = EventBatcher(batch_window_ms=100)

        batcher.add(tool_progress_event("scan", 0, 1, 100))
        batcher.add(tool_progress_event("scan", 0, 2, 100))

        batcher.clear()
        assert batcher.pending_count == 0

    def test_batcher_custom_window(self) -> None:
        """Batcher should respect custom batch window."""
        batcher = EventBatcher(batch_window_ms=500)

        # Should be 500ms / 1000 = 0.5 seconds
        assert batcher.batch_window == 0.5


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_first_event_emits(self) -> None:
        """First event of any type should emit."""
        limiter = RateLimiter(default_interval_ms=100)
        event = tool_progress_event("scan", 0, 1, 100)

        assert limiter.should_emit(event) is True

    def test_rate_limiter_blocks_rapid_events(self) -> None:
        """Rapid events of same type should be blocked."""
        limiter = RateLimiter(default_interval_ms=100)

        e1 = tool_progress_event("scan", 0, 1, 100)
        assert limiter.should_emit(e1) is True

        e2 = tool_progress_event("scan", 0, 2, 100)
        assert limiter.should_emit(e2) is False

    def test_rate_limiter_allows_after_interval(self) -> None:
        """Events should emit after interval passes."""
        limiter = RateLimiter(default_interval_ms=10)

        e1 = tool_progress_event("scan", 0, 1, 100)
        limiter.should_emit(e1)

        time.sleep(0.02)

        e2 = tool_progress_event("scan", 0, 2, 100)
        assert limiter.should_emit(e2) is True

    def test_rate_limiter_custom_interval(self) -> None:
        """Custom intervals should be respected."""
        limiter = RateLimiter()
        limiter.set_interval(SSEEventType.HEARTBEAT, 1000)  # 1 second

        e1 = message_event("assistant", "msg1")
        e2 = message_event("assistant", "msg2")

        # Both should emit with default interval (100ms is default)
        limiter.should_emit(e1)
        time.sleep(0.15)
        assert limiter.should_emit(e2) is True

    def test_rate_limiter_different_types_independent(self) -> None:
        """Different event types should have independent rate limits."""
        limiter = RateLimiter(default_interval_ms=100)

        # First progress
        progress = tool_progress_event("scan", 0, 1, 100)
        assert limiter.should_emit(progress) is True

        # Message should still emit (different type)
        msg = message_event("assistant", "Hello")
        assert limiter.should_emit(msg) is True

    def test_rate_limiter_reset_specific_type(self) -> None:
        """Reset should clear limit for specific type."""
        limiter = RateLimiter(default_interval_ms=100)

        e1 = tool_progress_event("scan", 0, 1, 100)
        limiter.should_emit(e1)

        # Immediate second would be blocked
        e2 = tool_progress_event("scan", 0, 2, 100)
        assert limiter.should_emit(e2) is False

        # Reset allows immediate emit
        limiter.reset(SSEEventType.TOOL_PROGRESS)
        e3 = tool_progress_event("scan", 0, 3, 100)
        assert limiter.should_emit(e3) is True

    def test_rate_limiter_reset_all(self) -> None:
        """Reset with no argument should clear all limits."""
        limiter = RateLimiter(default_interval_ms=100)

        limiter.should_emit(tool_progress_event("scan", 0, 1, 100))
        limiter.should_emit(message_event("assistant", "test"))

        limiter.reset()

        # Both should emit now
        assert limiter.should_emit(tool_progress_event("scan", 0, 2, 100)) is True
        assert limiter.should_emit(message_event("assistant", "test2")) is True


class TestDefaultRateLimiter:
    """Tests for default rate limiter configuration."""

    def test_default_rate_limits_defined(self) -> None:
        """Default rate limits should be defined for common types."""
        assert SSEEventType.TOOL_PROGRESS in DEFAULT_RATE_LIMITS
        assert SSEEventType.HEARTBEAT in DEFAULT_RATE_LIMITS
        assert SSEEventType.THINKING in DEFAULT_RATE_LIMITS

    def test_create_default_rate_limiter(self) -> None:
        """create_default_rate_limiter should create configured limiter."""
        limiter = create_default_rate_limiter()

        assert isinstance(limiter, RateLimiter)
        # Verify it has custom intervals set
        assert SSEEventType.TOOL_PROGRESS in limiter._intervals
