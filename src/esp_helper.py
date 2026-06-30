"""Screen reading and position tracking for fruit and event detection.

Uses OpenCV template matching and PIL for screen capture to locate
in-game objects such as devil fruits, NPCs, and sea event indicators.
This module is intended for educational demonstration purposes only.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import ImageGrab

logger = logging.getLogger("blox_fruits_tool.esp")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Holds the result of a single template-matching detection."""

    label: str               # Human-readable name (e.g. "Devil Fruit")
    confidence: float        # Match confidence 0.0 – 1.0
    center: Tuple[int, int]  # (x, y) center of the matched region
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)


# ── Main helper class ────────────────────────────────────────────────────────

class ESPHelper:
    """Screen-analysis assistant that locates in-game objects via template matching.

    Template images should be stored as PNG files inside a ``templates/``
    directory at the same level as this script.  Each file name (without
    extension) becomes the detection label.

    Usage::

        esp = ESPHelper(templates_dir="templates", confidence=0.8)
        results = esp.scan_screen()
        for r in results:
            print(f"{r.label} at {r.center} (conf={r.confidence:.2f})")
    """

    def __init__(
        self,
        templates_dir: str = "templates",
        confidence_threshold: float = 0.8,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        scale_steps: int = 5,
        highlight_color: Tuple[int, int, int] = (0, 255, 0),
        highlight_thickness: int = 2,
    ) -> None:
        """Initialise the ESP helper.

        Args:
            templates_dir: Path to directory containing template PNG images.
            confidence_threshold: Minimum confidence to consider a match valid.
            scale_range: (min_scale, max_scale) for multi-scale matching.
            scale_steps: Number of scale increments between min and max.
            highlight_color: BGR tuple for drawing bounding boxes.
            highlight_thickness: Line thickness for bounding boxes.
        """
        self._confidence = confidence_threshold
        self._scale_range = scale_range
        self._scale_steps = scale_steps
        self._color = highlight_color
        self._thickness = highlight_thickness
        self._templates: List[Tuple[str, np.ndarray]] = []
        self._load_templates(templates_dir)

    # -- template loading -----------------------------------------------------

    def _load_templates(self, directory: str) -> None:
        """Load all PNG images from *directory* as grayscale numpy arrays."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Templates directory not found: %s", directory)
            return
        for file_path in sorted(dir_path.glob("*.png")):
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    label = file_path.stem
                    self._templates.append((label, img))
                    logger.debug("Loaded template: %s (%dx%d)", label, *img.shape[::-1])
            except Exception as exc:
                logger.error("Failed to load template %s: %s", file_path, exc)
        logger.info("Loaded %d template(s) from %s.", len(self._templates), directory)

    # -- screen capture -------------------------------------------------------

    @staticmethod
    def capture_screen(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture a screenshot and return it as a BGR numpy array.

        Args:
            region: (left, top, width, height) or None for full screen.

        Returns:
            BGR image as a numpy array suitable for OpenCV operations.
        """
        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()
        frame = np.array(screenshot)
        # PIL gives RGB; convert to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -- matching -------------------------------------------------------------

    def _match_template(
        self,
        scene_gray: np.ndarray,
        template_gray: np.ndarray,
    ) -> Optional[Tuple[float, Tuple[int, int], Tuple[int, int, int, int]]]:
        """Perform multi-scale template matching.

        Returns:
            (confidence, center, bbox) or None if no match exceeds threshold.
        """
        th, tw = template_gray.shape[:2]
        sh, sw = scene_gray.shape[:2]
        if tw > sw or th > sh:
            return None

        scales = np.linspace(self._scale_range[0], self._scale_range[1], self._scale_steps)
        best_conf = 0.0
        best_loc = (0, 0)
        best_scale = 1.0

        for scale in scales:
            new_w = int(tw * scale)
            new_h = int(th * scale)
            if new_w < 5 or new_h < 5 or new_w > sw or new_h > sh:
                continue
            resized = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_conf:
                best_conf = max_val
                best_loc = max_loc
                best_scale = scale

        if best_conf < self._confidence:
            return None

        final_w = int(tw * best_scale)
        final_h = int(th * best_scale)
        cx = best_loc[0] + final_w // 2
        cy = best_loc[1] + final_h // 2
        bbox = (best_loc[0], best_loc[1], final_w, final_h)
        return (best_conf, (cx, cy), bbox)

    # -- public API -----------------------------------------------------------

    def scan_screen(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectionResult]:
        """Capture the screen and search for all loaded templates.

        Args:
            region: Optional (left, top, width, height) capture region.

        Returns:
            List of :class:`DetectionResult` for every match above threshold.
        """
        frame_bgr = self.capture_screen(region)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        results: List[DetectionResult] = []

        for label, template in self._templates:
            match = self._match_template(frame_gray, template)
            if match:
                conf, center, bbox = match
                results.append(DetectionResult(
                    label=label,
                    confidence=conf,
                    center=center,
                    bbox=bbox,
                ))
                logger.debug("Detected %s at %s (conf=%.3f)", label, center, conf)

        if results:
            logger.info("Scan complete: %d detection(s).", len(results))
        return results

    def draw_detections(
        self,
        frame: np.ndarray,
        results: List[DetectionResult],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of *frame*.

        Args:
            frame: BGR image array.
            results: Detection results from :meth:`scan_screen`.

        Returns:
            Annotated BGR image array (original is not modified).
        """
        annotated = frame.copy()
        for r in results:
            x, y, w, h = r.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), self._color, self._thickness)
            label_text = f"{r.label} {r.confidence:.0%}"
            cv2.putText(
                annotated,
                label_text,
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self._color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    def continuous_scan(
        self,
        interval: float = 0.5,
        callback: Optional[callable] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Run screen scans in a loop, calling *callback* with each result list.

        This method blocks until a KeyboardInterrupt is received.

        Args:
            interval: Seconds between consecutive scans.
            callback: Optional callable accepting a list of DetectionResult.
            region: Optional capture region.
        """
        logger.info("Starting continuous scan (interval=%.2f s).", interval)
        try:
            while True:
                results = self.scan_screen(region)
                if callback:
                    callback(results)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Continuous scan stopped by user.")
