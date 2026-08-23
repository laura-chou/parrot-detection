"""CLI execution entry point for local runs of Parrot YOLOv8 detector."""

import argparse
import logging
from src.detector import ParrotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parrot YOLOv8 Object Detection CLI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Inference mode: 'image' or 'video' (default: image)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="media/input/napping_image.jpg",
        help="Path to input image or video file (default: media/input/napping_image.jpg)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/parrot_yolov8.pt",
        help="Path to model weights file (default: models/parrot_yolov8.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (default: 0.5)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Window scaling factor for video preview display (default: 0.5)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV window during video processing (local GUI environment)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Initializing detector with weights: {args.weights}")
    detector = ParrotDetector(model_path=args.weights)

    if args.mode == "image":
        logger.info(f"Running image detection on source: {args.source}")
        _, output_path, detections, formatted_text = detector.detect_image(
            image_path=args.source, conf=args.conf
        )
        logger.info(f"Detection Results:\n{formatted_text}")
        logger.info(f"Saved annotated image to: {output_path}")

    elif args.mode == "video":
        logger.info(f"Running video detection on source: {args.source}")
        output_path = None
        for vid_out, status in detector.detect_video(
            video_path=args.source,
            conf=args.conf,
            scale=args.scale,
            show=args.show,
        ):
            if vid_out is not None:
                output_path = vid_out
            logger.info(status)
        logger.info(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
