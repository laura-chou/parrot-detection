"""Parrot object detection module using YOLOv8."""

import logging
import os
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import format_detections, format_detections_text, get_output_path, resize_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ParrotDetector:
    """Modular YOLOv8 Detector for Parrot Recognition."""

    def __init__(self, model_path: str = "models/parrot_yolov8.pt") -> None:
        """Initializes the ParrotDetector with model weights.

        :param model_path: Path to YOLOv8 weights file (.pt).
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            logger.warning(
                f"Model weights file not found at '{model_path}'. Model initialization may download or fail."
            )
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("Model loaded successfully.")

    def detect_image(
        self, image_path: str, conf: float = 0.5, output_dir: str = "media/output"
    ) -> Tuple[np.ndarray, str, List[Dict[str, Any]], str]:
        """Runs object detection on a single image.

        :param image_path: Path to the input image file.
        :param conf: Confidence threshold for detection.
        :param output_dir: Directory where the output image should be saved.
        :return: Tuple containing (annotated_image_rgb, output_path, detections_list, formatted_text).
        """
        logger.info(f"Starting image analysis: {image_path} (conf={conf})")
        if not os.path.exists(image_path):
            error_msg = f"Image path does not exist: {image_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        results = self.model(image_path, conf=conf)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        out_path = get_output_path(image_path, output_dir=output_dir, prefix="result_")
        cv2.imwrite(out_path, annotated_bgr)
        logger.info(f"Image analysis completed. Saved output to: {out_path}")

        detections = format_detections(results[0])
        formatted_text = format_detections_text(detections)

        return annotated_rgb, out_path, detections, formatted_text

    def detect_video(
        self,
        video_path: str,
        conf: float = 0.5,
        scale: float = 0.5,
        show: bool = False,
        output_dir: str = "media/output",
    ) -> str:
        """Runs object detection on a video stream or file.

        :param video_path: Path to input video file or camera index string.
        :param conf: Confidence threshold for detection.
        :param scale: Window scaling factor for OpenCV display (if show=True).
        :param show: Whether to display OpenCV window during processing.
        :param output_dir: Directory where processed video is saved.
        :return: Path to the saved annotated video file.
        """
        logger.info(f"Starting video analysis: {video_path} (conf={conf}, scale={scale})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Unable to open video source: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        out_path = get_output_path(
            video_path,
            output_dir=output_dir,
            prefix="result_",
        )
        base_name, _ = os.path.splitext(out_path)
        out_path = f"{base_name}.mp4"

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
                results = self.model(frame, conf=conf, verbose=False)
                annotated_frame = results[0].plot()

                writer.write(annotated_frame)

                if show:
                    preview_frame = resize_frame(annotated_frame, scale=scale)
                    cv2.imshow("Video Inference (Press 'q' to exit)", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User interrupted video display.")
                        break

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

        logger.info(f"Video analysis completed ({frame_count} frames processed). Saved to: {out_path}")
        return out_path
