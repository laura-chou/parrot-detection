"""Parrot object detection module using YOLOv8."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO

from src.utils import (
    format_detections,
    format_detections_text,
    get_output_path,
    reencode_to_h264,
    resize_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ParrotDetector:
    """Modular YOLOv8 Detector for Parrot Recognition and Behavior Analysis."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/parrot_behavior.pt",
        gatekeeper_path: Union[str, Path] = "models/parrot_detector.pt",
    ) -> None:
        """Initializes the ParrotDetector with model weights.

        :param model_path: Path to the main YOLOv8 behavior weights file (.pt).
        :param gatekeeper_path: Path to the custom YOLOv8 gatekeeper weights file (.pt).
        """
        self.model_path = str(model_path)
        self.gatekeeper_path = str(gatekeeper_path)

        # Initialize Main Model (Behavior Model)
        if not Path(self.model_path).exists():
            logger.warning(
                f"Main behavior model not found at '{self.model_path}'. Initialization may download or fail."
            )
        logger.info(f"Loading behavior model from: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Initialize Gatekeeper Model
        if not Path(self.gatekeeper_path).exists():
            logger.warning(
                f"Gatekeeper model not found at '{self.gatekeeper_path}'. Initialization may download or fail."
            )
        logger.info(f"Loading gatekeeper model from: {self.gatekeeper_path}")
        self.gatekeeper = YOLO(self.gatekeeper_path)

        logger.info("Both Main and Gatekeeper models loaded successfully.")

    def _clear_gpu_memory(self) -> None:
        """Utility method to clear PyTorch CUDA GPU cache if available."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has_parrot(self, source: Any, conf: float = 0.4) -> bool:
        """Private helper method checking if source contains a parrot (custom class ID 0).

        :param source: Image path, ndarray frame, etc.
        :param conf: Confidence threshold for gatekeeper model.
        :return: True if parrot detected, False otherwise.
        """
        results = self.gatekeeper(source, conf=conf, verbose=False)
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy()
                if 0 in classes:
                    return True
        return False

    def detect_image(
        self,
        image_path: Union[str, Path],
        conf: float = 0.5,
        output_dir: Union[str, Path] = "media/output",
    ) -> Tuple[np.ndarray, str, List[Dict[str, Any]], str]:
        """Runs object detection on a single image.

        :param image_path: Path to the input image file.
        :param conf: Confidence threshold for detection.
        :param output_dir: Directory where the output image should be saved.
        :return: Tuple containing (annotated_image_rgb, output_path, detections_list, formatted_text).
        """
        image_path_str = str(image_path)
        logger.info(f"Starting image analysis: {image_path_str} (conf={conf})")
        if not Path(image_path_str).exists():
            error_msg = f"Image path does not exist: {image_path_str}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        out_path = get_output_path(
            image_path_str, output_dir=str(output_dir), prefix=f"result_conf{conf:.2f}_"
        )

        if not self._has_parrot(image_path_str):
            logger.info("Gatekeeper: No parrot detected in image. Skipping main model.")
            original_bgr = cv2.imread(image_path_str)
            if original_bgr is not None:
                annotated_bgr = original_bgr.copy()
                cv2.putText(
                    annotated_bgr,
                    "No Parrot Detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_path, annotated_bgr)
            else:
                annotated_rgb = np.zeros((300, 300, 3), dtype=np.uint8)

            no_parrot_msg = "No parrot detected. Skipping analysis."
            return annotated_rgb, out_path, [], no_parrot_msg

        results = self.model(image_path_str, conf=conf)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(out_path, annotated_bgr)
        logger.info(f"Image analysis completed. Saved output to: {out_path}")

        detections = format_detections(results[0])
        formatted_text = format_detections_text(detections)

        self._clear_gpu_memory()
        return annotated_rgb, out_path, detections, formatted_text

    def detect_video(
        self,
        video_path: Union[str, Path, int],
        conf: float = 0.5,
        scale: float = 0.5,
        show: bool = False,
        output_dir: Union[str, Path] = "media/output",
    ) -> Generator[Tuple[Optional[str], str], None, None]:
        """Runs object detection on a video file as a generator streaming progress.

        :param video_path: Path to input video file or camera index string/int.
        :param conf: Confidence threshold for detection.
        :param scale: Window scaling factor for OpenCV display (if show=True).
        :param show: Whether to display OpenCV window during processing.
        :param output_dir: Directory where processed video is saved.
        :yields: Tuple of (video_output_path_or_None, status_message_string)
        """
        logger.info(f"Starting video analysis: {video_path} (conf={conf}, scale={scale})")

        # Convert video_path to int if it represents a camera index
        video_source: Any = video_path
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            video_source = int(video_path)
            sample_name = f"webcam_{video_source}.mp4"
        else:
            sample_name = str(video_path)

        out_path = get_output_path(
            sample_name,
            output_dir=str(output_dir),
            prefix=f"result_conf{conf:.2f}_",
        )
        base_name, _ = os.path.splitext(out_path)
        out_path = f"{base_name}.mp4"

        # Check if output video already exists to avoid redundant processing
        if isinstance(video_source, str) and Path(out_path).exists():
            logger.info(f"Output video already exists at '{out_path}'. Skipping detection.")
            yield out_path, "Loaded cached detection video."
            return

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            error_msg = f"Unable to open video source: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        yield None, f"Processing video: 0/{total_frames if total_frames > 0 else '?'} frames (0.0%)..."

        # Try standard fourcc video encoders for cross-platform compatibility
        codecs = ["mp4v", "avc1", "XVID"]
        writer = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if writer.isOpened():
                break

        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logger.info("Video frame stream ended.")
                    break

                frame_count += 1
                if self._has_parrot(frame):
                    results = self.model(frame, conf=conf, verbose=False)
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(
                        annotated_frame,
                        "No Parrot Detected",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(annotated_frame)

                if show:
                    preview_frame = resize_frame(annotated_frame, scale=scale)
                    cv2.imshow("Video Inference (Press 'q' to exit)", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User interrupted video display.")
                        break

                # Yield progress periodically (every 30 frames or on last frame)
                if frame_count % 30 == 0 or (total_frames > 0 and frame_count == total_frames):
                    progress_pct = (
                        (frame_count / total_frames) * 100 if total_frames > 0 else 0.0
                    )
                    status_msg = (
                        f"Processing video: {frame_count}/{total_frames if total_frames > 0 else '?'} "
                        f"frames ({progress_pct:.1f}%)..."
                    )
                    yield None, status_msg

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._clear_gpu_memory()

        logger.info(f"Video analysis completed ({frame_count} frames processed). Saved to: {out_path}")

        yield None, "Re-encoding video for web playback compatibility..."

        # Re-encode video to browser-compatible H.264 format
        out_path = reencode_to_h264(out_path)

        summary_msg = f"Analysis complete! {frame_count} frames processed. Video ready."
        yield out_path, summary_msg
