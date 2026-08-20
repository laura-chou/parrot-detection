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
    with gr.Blocks(title="Parrot Detection YOLOv8 System") as demo:
        gr.Markdown(
            """
            # 🦜 Parrot Recognition & Object Detection
            Modular Object Detection application powered by **YOLOv8** and **Gradio**.
            """
        )

        with gr.Tabs():
            # Tab 1: Image Analysis
            with gr.TabItem("Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(
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

                        sample_img = "media/input/napping_image.jpg"
                        if os.path.exists(sample_img):
                            gr.Examples(
                                examples=[[sample_img, 0.5]],
                                inputs=[img_input, img_conf],
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

                        sample_vid = "media/input/eating_video.mp4"
                        if os.path.exists(sample_vid):
                            gr.Examples(
                                examples=[[sample_vid, 0.5]],
                                inputs=[vid_input, vid_conf],
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
