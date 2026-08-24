"""Gradio web application interface for Pacific Parrotlet Recognition using YOLOv8."""

import logging
import os
import gradio as gr
from src.detector import ParrotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/parrot_yolov8.pt"
detector = ParrotDetector(model_path=MODEL_PATH)


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
    default_image = "media/input/napping_image.jpg"
    default_video = "media/input/compilation_video.mp4"
    fallback_video = "media/input/eating_video.mp4"

    if not os.path.exists(default_video) and os.path.exists(fallback_video):
        active_video = fallback_video
    else:
        active_video = default_video

    with gr.Blocks(title="Pacific Parrotlet Behavior & Object Detection System") as demo:
        gr.Markdown(
            """
            # 🦜 Pacific Parrotlet Behavior & Object Detection System
            A YOLOv8-powered deep learning model capable of detecting 14 distinct Pacific Parrotlet behaviors and actions (e.g., eating, napping, preening, stretching, scouting, excited).

            *Note: This model is fine-tuned specifically for Pacific Parrotlets and is not suitable for other bird species.*
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
                            sources=["webcam", "upload"],
                            height=360,
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
                            ["media/input/sleepy.jpg", 0.5],
                            ["media/input/scouting.jpg", 0.5],
                            ["media/input/gnawing.jpg", 0.5],
                        ]
                        gr.Examples(
                            examples=image_examples,
                            inputs=[img_input, img_conf],
                            label="Click an example below to test image detection:",
                        )

                    with gr.Column():
                        img_output = gr.Image(label="Annotated Result", height=360)
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
                            value=active_video if os.path.exists(active_video) else None,
                            label="Upload Video or Record",
                            sources=["webcam", "upload"],
                            height=360,
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
                            ["media/input/compilation_video.mp4", 0.5],
                        ]
                        gr.Examples(
                            examples=video_examples,
                            inputs=[vid_input, vid_conf],
                            label="Click an example below to test behavior detection:",
                        )

                    with gr.Column():
                        vid_output = gr.Video(
                            label="Processed Output Video (Preview & Download)",
                            height=360,
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


demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
