import gradio as gr
from PIL import Image
import sys
import yaml
import os
import numpy as np
import torch
from torchvision.models import resnet50

# Append the path to the src/xai_method directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'xai_methods'))

# Import the necessary functions from cam.py and the DFF script
from cam import initModel, processImage, visualize_image as visualize_gradcam
from dff import visualize_image as visualize_dff  # Assuming your DFF script is named dff.py

# Load configuration
with open("./CONFIG.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def gradio_interface(image_path, method, n_components=5, top_k=2):
    if method in ["gradcam", "hirescam"]:
        # Call the visualize_image function from cam.py for Grad-CAM and HiResCAM
        result_image = visualize_gradcam(image_path, method)
    elif method == "dff":
        # Call the visualize_image function from dff.py for Deep Feature Factorization
        model = resnet50(pretrained=True)
        model.eval()
        result_image = visualize_dff(model, image_path, n_components, top_k)
    else:
        raise ValueError("Invalid method. Use 'gradcam', 'hirescam', or 'dff'.")

    return result_image

image_width = 400
image_height = 400

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload Image", width=image_width, height=image_height),
        gr.Radio(choices=["gradcam", "hirescam", "dff"], label="Method"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Components (DFF only)"),
        gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Top K (DFF only)")
    ],
    outputs=gr.Image(type="pil", label="Output Image", width=image_width, height=image_height),
    title="Explainable AI for Image Classification",
    description="Upload an image, select a method (Grad-CAM, HiResCAM, or DFF), and specify the output filename to generate the explanation overlay.",
)

if __name__ == "__main__":
    iface.launch()
