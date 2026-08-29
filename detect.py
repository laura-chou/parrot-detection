"""CLI execution entry point for local runs of Pacific Parrotlet YOLOv8 detector."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

from src.detector import ParrotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the CLI detection interface."""
    parser = argparse.ArgumentParser(
        description="Pacific Parrotlet YOLOv8 Object & Behavior Detection CLI"
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
        default="media/input/napping.jpg",
        help="Path to input image or video file, or webcam index (e.g. 0) (default: media/input/napping.jpg)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/parrot_behavior.pt",
        help="Path to main behavior model weights file (default: models/parrot_behavior.pt)",
    )
    parser.add_argument(
        "--gatekeeper",
        type=str,
        default="models/parrot_detector.pt",
        help="Path to gatekeeper detector weights file (default: models/parrot_detector.pt)",
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


def main() -> None:
    """Main execution function for CLI detection."""
    args = parse_args()
    logger.info(
        f"Initializing detector with weights: '{args.weights}' and gatekeeper: '{args.gatekeeper}'"
    )

    try:
        detector = ParrotDetector(model_path=args.weights, gatekeeper_path=args.gatekeeper)
    except Exception as e:
        logger.error(f"Failed to initialize ParrotDetector: {e}")
        sys.exit(1)

    if args.mode == "image":
        logger.info(f"Running image detection on source: {args.source}")
        if not Path(args.source).exists():
            logger.error(f"Specified input image source does not exist: {args.source}")
            sys.exit(1)

        try:
            _, output_path, detections, formatted_text = detector.detect_image(
                image_path=args.source, conf=args.conf
            )
            logger.info(f"Detection Results:\n{formatted_text}")
            logger.info(f"Saved annotated image to: {output_path}")
        except Exception as e:
            logger.error(f"Error during CLI image detection: {e}")
            sys.exit(1)

    elif args.mode == "video":
        logger.info(f"Running video detection on source: {args.source}")
        output_path: Optional[str] = None
        try:
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
        except Exception as e:
            logger.error(f"Error during CLI video detection: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
