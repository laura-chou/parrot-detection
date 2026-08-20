"""Utility functions for file paths, frame resizing, result formatting, and video encoding."""

import logging
import os
import subprocess
from typing import Any, Dict, List
import cv2
import imageio_ffmpeg
import numpy as np

logger = logging.getLogger(__name__)


def get_output_path(
    input_path: str, output_dir: str = "media/output", prefix: str = "result_"
) -> str:
    """Generates an output file path inside output_dir based on input_path.

    :param input_path: Path to the input file.
    :param output_dir: Directory where the output file should be saved.
    :param prefix: Prefix added to the original filename.
    :return: Full path for the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    if not filename:
        filename = "output.jpg"
    out_filename = f"{prefix}{filename}"
    return os.path.join(output_dir, out_filename)


def resize_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Resizes an image or video frame by a scaling factor.

    :param frame: Input image frame as a NumPy array.
    :param scale: Scaling factor (e.g., 0.5 for 50% scale).
    :return: Resized frame.
    """
    if scale <= 0 or scale == 1.0:
        return frame
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)


def format_detections(results: Any) -> List[Dict[str, Any]]:
    """Formats YOLO prediction results into a list of detection dictionaries.

    Each item in the returned list contains class_id, class_name,
    confidence score (float), and formatted percentage string.

    :param results: YOLO prediction results object list or single result object.
    :return: List of detection summaries.
    """
    detections: List[Dict[str, Any]] = []

    if isinstance(results, list):
        results_list = results
    else:
        results_list = [results]

    for res in results_list:
        boxes = res.boxes
        if boxes is None:
            continue

        names = res.names
        for box in boxes:
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_name = names.get(cls_id, f"class_{cls_id}") if names else f"class_{cls_id}"
            conf_percent = f"{conf * 100:.2f}%"

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "confidence_percentage": conf_percent,
                }
            )

    return detections


def format_detections_text(detections: List[Dict[str, Any]]) -> str:
    """Formats detection dictionary list into human-readable text.

    :param detections: List of detection dictionaries.
    :return: Formatted string for UI display.
    """
    if not detections:
        return "No objects detected."

    lines = []
    for idx, det in enumerate(detections, start=1):
        lines.append(f"{idx}. {det['class_name']}: {det['confidence_percentage']}")
    return "\n".join(lines)


def reencode_to_h264(video_path: str) -> str:
    """Re-encodes a video file to H.264 (yuv420p) format using FFmpeg binary from imageio-ffmpeg.

    This ensures full HTML5 video browser compatibility in Gradio web interfaces.

    :param video_path: Path to the input video file.
    :return: Path to the re-encoded video file, or original video_path if conversion fails.
    """
    if not os.path.exists(video_path):
        logger.warning(f"Video path does not exist for re-encoding: {video_path}")
        return video_path

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        dir_name, file_name = os.path.split(video_path)
        temp_out_path = os.path.join(dir_name, f"h264_{file_name}")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            video_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            temp_out_path,
        ]

        logger.info(f"Re-encoding video to H.264 using FFmpeg: {video_path}")
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        logger.info("FFmpeg re-encoding completed successfully.")

        os.replace(temp_out_path, video_path)
        return video_path

    except Exception as e:
        logger.error(
            f"Failed to re-encode video '{video_path}' to H.264: {e}. Falling back to original video."
        )
        if 'temp_out_path' in locals() and os.path.exists(temp_out_path):
            try:
                os.remove(temp_out_path)
            except Exception:
                pass
        return video_path
