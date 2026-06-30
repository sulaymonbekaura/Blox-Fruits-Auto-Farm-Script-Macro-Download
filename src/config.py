"""Configuration settings for the Blox Fruits automation tool.

All values can be adjusted at runtime or by modifying this file directly.
Each setting includes a type hint and a brief explanation.
"""


class Config:
    """Central configuration container for the automation tool."""

    # ── Hotkey bindings ──────────────────────────────────────────────────
    HOTKEY_START: str = "f5"          # Start the macro loop
    HOTKEY_STOP: str = "f6"           # Stop the macro loop
    HOTKEY_RECORD: str = "f7"         # Begin recording mouse/keyboard events
    HOTKEY_REPLAY: str = "f8"         # Replay the last recorded macro
    HOTKEY_TOGGLE_ESP: str = "f9"     # Enable / disable the ESP overlay

    # ── Timing delays (seconds) ──────────────────────────────────────────
    CLICK_DELAY: float = 0.1          # Pause between simulated clicks
    KEY_DELAY: float = 0.05           # Pause between simulated key presses
    LOOP_DELAY: float = 0.5           # Pause between macro loop iterations
    SCREEN_CAPTURE_INTERVAL: float = 0.5  # Seconds between ESP screen reads

    # ── Macro behaviour ──────────────────────────────────────────────────
    MAX_RECORDING_DURATION: int = 300  # Maximum recording length in seconds
    LOOP_COUNT: int = 0               # Number of loops (0 = infinite)
    MOUSE_SENSITIVITY: float = 1.0    # Multiplier for recorded mouse deltas

    # ── ESP / screen analysis ────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.8 # Minimum match confidence (0.0 – 1.0)
    TEMPLATE_SCALE_RANGE: tuple = (0.8, 1.2)  # Min/max scale for template matching
    ESP_HIGHLIGHT_COLOR: tuple = (0, 255, 0)   # BGR colour for ESP overlay
    ESP_HIGHLIGHT_THICKNESS: int = 2

    # ── Window targeting ─────────────────────────────────────────────────
    TARGET_WINDOW_TITLE: str = "Blox Fruits"
    CAPTURE_REGION: tuple = None       # (left, top, width, height) or None for full screen

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"            # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "blox_fruits_tool.log"
