"""Macro recorder and player for automating mouse and keyboard sequences.

Provides the :class:`MacroRecorder` to capture user input events and the
:class:`MacroPlayer` to replay them with configurable timing and looping.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import logging

import pyautogui
import keyboard

logger = logging.getLogger("blox_fruits_tool.macro")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MacroEvent:
    """Represents a single recorded input event."""

    timestamp: float           # Relative time in seconds since recording started
    event_type: str            # One of: 'key_down', 'key_up', 'mouse_click', 'mouse_move'
    data: dict = field(default_factory=dict)
    # data examples:
    #   key_down / key_up   -> {"key": "a"}
    #   mouse_click         -> {"button": "left", "x": 500, "y": 300}
    #   mouse_move          -> {"x": 510, "y": 305}


# ── Recorder ─────────────────────────────────────────────────────────────────

class MacroRecorder:
    """Records keyboard and mouse events into a list of :class:`MacroEvent` objects.

    Usage::

        recorder = MacroRecorder()
        recorder.start()
        # ... user performs actions ...
        recorder.stop()
        events = recorder.get_events()
    """

    def __init__(self, max_duration: int = 300) -> None:
        """Initialise the recorder.

        Args:
            max_duration: Maximum recording length in seconds (0 = unlimited).
        """
        self._events: List[MacroEvent] = []
        self._start_time: float = 0.0
        self._recording: bool = False
        self._max_duration: int = max_duration
        self._lock: threading.Lock = threading.Lock()
        self._hooks: list = []

    # -- keyboard callbacks ---------------------------------------------------

    def _on_key_down(self, event: keyboard.KeyboardEvent) -> None:
        if not self._recording:
            return
        if self._max_duration and (time.time() - self._start_time) > self._max_duration:
            self.stop()
            return
        with self._lock:
            self._events.append(MacroEvent(
                timestamp=time.time() - self._start_time,
                event_type="key_down",
                data={"key": event.name},
            ))

    def _on_key_up(self, event: keyboard.KeyboardEvent) -> None:
        if not self._recording:
            return
        with self._lock:
            self._events.append(MacroEvent(
                timestamp=time.time() - self._start_time,
                event_type="key_up",
                data={"key": event.name},
            ))

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Begin recording input events."""
        if self._recording:
            logger.warning("Recorder is already running.")
            return
        self._events.clear()
        self._start_time = time.time()
        self._recording = True
        self._hooks.append(keyboard.hook(self._on_key_down))
        self._hooks.append(keyboard.hook(self._on_key_up))
        logger.info("Recording started (max %d s).", self._max_duration or 0)

    def stop(self) -> None:
        """Stop recording and release keyboard hooks."""
        if not self._recording:
            return
        self._recording = False
        for hook in self._hooks:
            keyboard.unhook(hook)
        self._hooks.clear()
        logger.info("Recording stopped. %d events captured.", len(self._events))

    def get_events(self) -> List[MacroEvent]:
        """Return a copy of the recorded events list."""
        with self._lock:
            return list(self._events)

    @property
    def is_recording(self) -> bool:
        return self._recording


# ── Player ───────────────────────────────────────────────────────────────────

class MacroPlayer:
    """Replays a list of :class:`MacroEvent` objects with optional looping.

    Usage::

        player = MacroPlayer(events, loop_count=0)
        player.play()   # blocks until stopped or finished
    """

    def __init__(
        self,
        events: List[MacroEvent],
        loop_count: int = 0,
        click_delay: float = 0.1,
        key_delay: float = 0.05,
        loop_delay: float = 0.5,
        speed_multiplier: float = 1.0,
    ) -> None:
        """Initialise the player.

        Args:
            events: List of MacroEvent objects to replay.
            loop_count: Number of playback loops (0 = infinite).
            click_delay: Extra delay after mouse clicks (seconds).
            key_delay: Extra delay after key presses (seconds).
            loop_delay: Pause between loop iterations (seconds).
            speed_multiplier: Playback speed (1.0 = normal, 2.0 = double speed).
        """
        self._events = events
        self._loop_count = loop_count
        self._click_delay = click_delay
        self._key_delay = key_delay
        self._loop_delay = loop_delay
        self._speed = speed_multiplier
        self._playing: bool = False
        self._stop_event = threading.Event()

    def _replay_once(self) -> None:
        """Execute a single pass through the event list."""
        prev_timestamp = 0.0
        for evt in self._events:
            if self._stop_event.is_set():
                return
            # Wait for the appropriate amount of time
            wait = (evt.timestamp - prev_timestamp) / self._speed
            if wait > 0:
                time.sleep(wait)
            prev_timestamp = evt.timestamp

            # Execute the event
            try:
                if evt.event_type == "key_down":
                    keyboard.press(evt.data["key"])
                    time.sleep(self._key_delay)
                elif evt.event_type == "key_up":
                    keyboard.release(evt.data["key"])
                    time.sleep(self._key_delay)
                elif evt.event_type == "mouse_click":
                    pyautogui.click(
                        x=evt.data.get("x"),
                        y=evt.data.get("y"),
                        button=evt.data.get("button", "left"),
                    )
                    time.sleep(self._click_delay)
                elif evt.event_type == "mouse_move":
                    pyautogui.moveTo(
                        x=evt.data["x"],
                        y=evt.data["y"],
                        duration=0.0,
                    )
            except Exception as exc:
                logger.error("Error replaying event %s: %s", evt, exc)

    def play(self) -> None:
        """Start playback (blocks the calling thread until finished or stopped)."""
        if self._playing:
            logger.warning("Player is already running.")
            return
        if not self._events:
            logger.warning("No events to replay.")
            return

        self._playing = True
        self._stop_event.clear()
        logger.info("Playback started (%d loops, speed %.1fx).",
                     self._loop_count or float("inf"), self._speed)

        iteration = 0
        while self._playing:
            if self._loop_count and iteration >= self._loop_count:
                break
            self._replay_once()
            if self._stop_event.is_set():
                break
            iteration += 1
            if self._loop_count == 0 or iteration < self._loop_count:
                time.sleep(self._loop_delay)

        self._playing = False
        logger.info("Playback stopped after %d iteration(s).", iteration)

    def stop(self) -> None:
        """Signal the player to stop after the current event."""
        self._stop_event.set()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
