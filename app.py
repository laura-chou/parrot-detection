"""Gradio web application interface for Pacific Parrotlet Recognition using YOLOv8."""

# 1. 標準庫 (內建)
import logging
import os
import sys
import asyncio
from asyncio.proactor_events import _ProactorBasePipeTransport
from functools import wraps

# 2. 第三方套件
import gradio as gr
import spaces

# 3. 本地專案自訂模組
from src.detector import ParrotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/parrot_behavior.pt"
GATEKEEPER_PATH = "models/parrot_detector.pt"
detector = ParrotDetector(model_path=MODEL_PATH, gatekeeper_path=GATEKEEPER_PATH)

# 【Windows 專用】解決 Windows 關閉或重整網頁時，終端機「遠端主機已強制關閉連線 (WinError 10054)」的錯誤。
# 做法：攔截 asyncio 底層 Proactor 管道的連線中斷回呼，將該特定異常靜音（Pass），
# 並手動釋放 socket 資源，以防程式關閉時觸發「Invalid file descriptor: -1」的垃圾回收報錯。
if sys.platform == "win32":
    def silence_connection_lost(func):

        @wraps(func)
        def wrapper(self, exc=None):
            try:
                return func(self, exc)
            except (ConnectionResetError, AttributeError):
                self._sock = None
        return wrapper

    _ProactorBasePipeTransport._call_connection_lost = silence_connection_lost(_ProactorBasePipeTransport._call_connection_lost)

def create_windows_loop():
    """建立標準的 Windows 事件迴圈"""
    return asyncio.new_event_loop()

@spaces.GPU
def process_image(image_path: str, conf_threshold: float):
    """Callback function for Image Analysis tab in Gradio."""
    if not image_path:
        return None, "Please upload an image."

    try:
        annotated_rgb, _, detections, formatted_text = detector.detect_image(
            image_path=image_path, conf=conf_threshold
        )
        return annotated_rgb, formatted_text
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return None, f"Error: {str(e)}"

@spaces.GPU
def process_video(video_path: str, conf_threshold: float):
    """Callback function for Video Analysis tab in Gradio using a generator."""
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


def create_ui():
    """Builds and configures the Gradio Blocks web interface."""
    default_image = "media/input/napping.jpg"
    default_video = "media/input/compilation_video.mp4"

    with gr.Blocks(title="Pacific Parrotlet Behavior & Object Detection System") as demo:
        gr.Markdown(
            """
            # 🦜 Pacific Parrotlet Behavior & Object Detection System
            A YOLOv8-powered deep learning model capable of detecting 13 distinct Pacific Parrotlet behaviors and actions (e.g., eating, drinking, napping, preening, stretching, observing).

            *Note: Behavior detection is fine-tuned specifically for Pacific Parrotlets; accuracy may decrease when applied to other bird species.*
            """
        )

        with gr.Tabs():
            # Tab 1: Image Analysis
            with gr.TabItem("Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(
                            value=default_image if os.path.exists(default_image) else None,
                            type="filepath",
                            label="Upload Image or Take Photo",
                            sources=["upload", "webcam"],
                            height=480
                        )
                        img_conf = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
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
                        img_output = gr.Image(label="Annotated Result", height=480)
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

            # Tab 2: Video Analysis
            with gr.TabItem("Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(
                            value=default_video if os.path.exists(default_video) else None,
                            label="Upload Video or Record",
                            sources=["upload", "webcam"]
                        )
                        vid_conf = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold",
                        )
                        vid_button = gr.Button("Analyze Video", variant="primary")

                        video_examples = [
                            ["media/input/compilation_video.mp4", 0.5]
                        ]
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
