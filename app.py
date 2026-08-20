"""Gradio web application interface for Parrot Recognition using YOLOv8."""

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
        return None, "Please upload an image.", []

    try:
        annotated_rgb, _, detections, formatted_text = detector.detect_image(
            image_path=image_path, conf=conf_threshold
        )
        return annotated_rgb, formatted_text, detections
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return None, f"Error: {str(e)}", []


def process_video(video_path: str, conf_threshold: float):
    """Callback function for Video Analysis tab in Gradio."""
    if not video_path:
        return None

    try:
        output_video_path = detector.detect_video(
            video_path=video_path, conf=conf_threshold, show=False
        )
        return output_video_path
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        return None


def create_ui():
    """Builds and configures the Gradio Blocks web interface."""
    default_image = "media/input/napping_image.jpg"
    default_video = "media/input/eating_video.mp4"

    with gr.Blocks(title="Parrot Behavior & Object Detection System") as demo:
        gr.Markdown(
            """
            # 🦜 Parrot Behavior & Object Detection System
            A YOLOv8-powered deep learning model capable of detecting 14 distinct parrot behaviors and actions (e.g., eating, napping, preening, stretching, scouting, excited).
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
                            label="Upload Image",
                            sources=["upload"],
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
                            ["media/input/napping_image.jpg", 0.5],
                            ["media/input/relaxed_image.jpg", 0.5],
                        ]
                        gr.Examples(
                            examples=image_examples,
                            inputs=[img_input, img_conf],
                            label="Click an example below to test image detection:",
                        )

                    with gr.Column():
                        img_output = gr.Image(label="Annotated Result")
                        det_text_output = gr.Textbox(
                            label="Detections (Class & Percentage)",
                            lines=5,
                            interactive=False,
                        )
                        det_json_output = gr.JSON(label="Detailed Detections Data")

                img_button.click(
                    fn=process_image,
                    inputs=[img_input, img_conf],
                    outputs=[img_output, det_text_output, det_json_output],
                )

            # Tab 2: Video Analysis
            with gr.TabItem("Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(
                            value=default_video if os.path.exists(default_video) else None,
                            label="Upload Video",
                            sources=["upload"],
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
                            ["media/input/eating_video.mp4", 0.5],
                            ["media/input/preening_video.mp4", 0.5],
                            ["media/input/stretching_video.mp4", 0.5],
                            ["media/input/excited_video.mp4", 0.5],
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

                vid_button.click(
                    fn=process_video,
                    inputs=[vid_input, vid_conf],
                    outputs=[vid_output],
                )

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
