"""Gradio web application interface for Pacific Parrotlet Recognition using YOLOv8."""

import asyncio
from asyncio.proactor_events import _ProactorBasePipeTransport
from dataclasses import dataclass
from functools import wraps
import logging
import os
from pathlib import Path
import sys
from typing import Any, Generator, Optional, Tuple

import gradio as gr

# Safe import for ZeroGPU compatibility on Hugging Face Spaces
try:
    import spaces

    GPU_DECORATOR = spaces.GPU
except ImportError:
    # Dummy decorator fallback when spaces package is not installed (e.g., local execution)
    def dummy_decorator(func):
        return func

    GPU_DECORATOR = dummy_decorator

from src.detector import ParrotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Centralized configuration constants for the Gradio application."""

    MODEL_PATH: str = "models/parrot_behavior.pt"
    GATEKEEPER_PATH: str = "models/parrot_detector.pt"
    DEFAULT_IMAGE: str = "media/input/napping.jpg"
    DEFAULT_VIDEO: str = "media/input/compilation_video.mp4"
    DEFAULT_CONFIDENCE: float = 0.5
    CONF_MIN: float = 0.1
    CONF_MAX: float = 1.0
    CONF_STEP: float = 0.05
    IMAGE_HEIGHT: int = 480


def custom_unraisablehook(unraisable: Any) -> None:
    """Custom unraisable exception hook to suppress harmless teardown errors."""
    if unraisable.exc_value is not None and "Invalid file descriptor: -1" in str(unraisable.exc_value):
        logger.info("Ignored 'Invalid file descriptor: -1' during teardown.")
    else:
        sys.__unraisablehook__(unraisable)


def setup_platform_compatibility() -> None:
    """Sets up cross-platform exception hooks and event loop workarounds."""
    # Global unraisable exception hook for garbage collection teardown issues
    sys.unraisablehook = custom_unraisablehook

    # [Windows-specific] Fixes terminal error "WinError 10054" upon browser tab close/refresh.
    if sys.platform == "win32":

        def silence_connection_lost(func):
            @wraps(func)
            def wrapper(self, exc=None):
                try:
                    return func(self, exc)
                except (ConnectionResetError, AttributeError):
                    self._sock = None

            return wrapper

        _ProactorBasePipeTransport._call_connection_lost = silence_connection_lost(
            _ProactorBasePipeTransport._call_connection_lost
        )


def create_windows_loop() -> asyncio.AbstractEventLoop:
    """Creates a standard Windows event loop."""
    return asyncio.new_event_loop()


# Initialize platform compatibility handlers
setup_platform_compatibility()

# Instantiate detector instance
detector = ParrotDetector(
    model_path=AppConfig.MODEL_PATH,
    gatekeeper_path=AppConfig.GATEKEEPER_PATH,
)


@GPU_DECORATOR
def process_image(image_path: Optional[str], conf_threshold: float) -> Tuple[Optional[Any], str]:
    """Callback function for Image Analysis tab in Gradio.

    :param image_path: Path to input image or None.
    :param conf_threshold: Detection confidence threshold.
    :return: Tuple of (annotated_rgb_image_or_None, formatted_status_text).
    """
    if not image_path:
        return None, "Please upload an image."

    try:
        annotated_rgb, _, _, formatted_text = detector.detect_image(
            image_path=image_path, conf=conf_threshold
        )
        return annotated_rgb, formatted_text
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return None, f"Error: {str(e)}"


@GPU_DECORATOR
def process_video(
    video_path: Optional[str], conf_threshold: float
) -> Generator[Tuple[Optional[str], str], None, None]:
    """Callback function for Video Analysis tab in Gradio using a generator.

    :param video_path: Path to input video or None.
    :param conf_threshold: Detection confidence threshold.
    :yields: Tuple of (video_output_path_or_None, status_message_text).
    """
    if not video_path:
        yield None, "Please upload a video file."
        return

    try:
        for video_update, text_update in detector.detect_video(
            video_path=video_path, conf=conf_threshold, show=False
        ):
            yield video_update, text_update
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        yield None, f"Error: {str(e)}"


def build_image_tab() -> None:
    """Constructs the Image Analysis tab UI components."""
    with gr.TabItem("Image Analysis"):
        with gr.Row():
            with gr.Column():
                default_img_path = (
                    AppConfig.DEFAULT_IMAGE
                    if Path(AppConfig.DEFAULT_IMAGE).exists()
                    else None
                )
                img_input = gr.Image(
                    value=default_img_path,
                    type="filepath",
                    label="Upload Image or Take Photo",
                    sources=["upload", "webcam"],
                    height=AppConfig.IMAGE_HEIGHT,
                )
                img_conf = gr.Slider(
                    minimum=AppConfig.CONF_MIN,
                    maximum=AppConfig.CONF_MAX,
                    value=AppConfig.DEFAULT_CONFIDENCE,
                    step=AppConfig.CONF_STEP,
                    label="Confidence Threshold",
                )
                img_button = gr.Button("Analyze Image", variant="primary")

                image_examples = [
                    ["media/input/sleeping.jpg", 0.5],
                    ["media/input/observing.jpg", 0.5],
                    ["media/input/napping.jpg", 0.5],
                ]
                gr.Examples(
                    examples=image_examples,
                    inputs=[img_input, img_conf],
                    label="Click an example below to test image detection:",
                )

            with gr.Column():
                img_output = gr.Image(
                    label="Annotated Result", height=AppConfig.IMAGE_HEIGHT
                )
                det_text_output = gr.Textbox(
                    label="Detections (Class & Percentage)",
                    lines=5,
                    interactive=False,
                )

        img_button.click(
            fn=process_image,
            inputs=[img_input, img_conf],
            outputs=[img_output, det_text_output],
        )


def build_video_tab() -> None:
    """Constructs the Video Analysis tab UI components."""
    with gr.TabItem("Video Analysis"):
        with gr.Row():
            with gr.Column():
                default_vid_path = (
                    AppConfig.DEFAULT_VIDEO
                    if Path(AppConfig.DEFAULT_VIDEO).exists()
                    else None
                )
                vid_input = gr.Video(
                    value=default_vid_path,
                    label="Upload Video or Record",
                    sources=["upload", "webcam"],
                )
                vid_conf = gr.Slider(
                    minimum=AppConfig.CONF_MIN,
                    maximum=AppConfig.CONF_MAX,
                    value=AppConfig.DEFAULT_CONFIDENCE,
                    step=AppConfig.CONF_STEP,
                    label="Confidence Threshold",
                )
                vid_button = gr.Button("Analyze Video", variant="primary")

                video_examples = [["media/input/compilation_video.mp4", 0.5]]
                gr.Examples(
                    examples=video_examples,
                    inputs=[vid_input, vid_conf],
                    label="Click an example below to test behavior detection:",
                )

            with gr.Column():
                vid_output = gr.Video(
                    label="Processed Output Video (Preview & Download)"
                )
                vid_status_output = gr.Textbox(
                    label="Processing Status & Progress",
                    lines=2,
                    interactive=False,
                )

        vid_button.click(
            fn=process_video,
            inputs=[vid_input, vid_conf],
            outputs=[vid_output, vid_status_output],
        )


def create_ui() -> gr.Blocks:
    """Builds and configures the Gradio Blocks web interface."""
    with gr.Blocks(title="Pacific Parrotlet Behavior & Object Detection System") as demo:
        gr.Markdown(
            """
            # 🦜 Pacific Parrotlet Behavior & Object Detection System
            A YOLOv8-powered deep learning model capable of detecting 13 distinct Pacific Parrotlet behaviors and actions (e.g., eating, drinking, napping, preening, stretching, observing).

            *Note: Behavior detection is fine-tuned specifically for Pacific Parrotlets; accuracy may decrease when applied to other bird species.*
            """
        )

        with gr.Tabs():
            build_image_tab()
            build_video_tab()

    return demo


if __name__ == "__main__":
    demo = create_ui()

    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=create_windows_loop) as runner:
            try:
                demo.queue().launch(ssr_mode=False)
            except KeyboardInterrupt:
                print("\nShutting down Gradio server...")
    else:
        demo.queue().launch(ssr_mode=False)
