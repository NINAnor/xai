import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sys
import yaml
import os
import argparse
import requests

from _utils import processImage

# Load configuration
with open("./CONFIG.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def initModel(pretrained=True):
    # Load pre-trained model and set to evaluation mode
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    model.eval()
    return model

def create_labels(concept_scores, top_k=2):
    """ Create a list with the ImageNet category names of the top scoring categories"""
    imagenet_categories_url = (
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/"
        "238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    )
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk

def visualize_image(image_path, method):
    model = initModel(pretrained=True)

    # Load and preprocess image
    image = Image.open(image_path)
    input_tensor = processImage(image)

    # Hook for gradients and activations
    gradients = []
    activations = []

    def hook_function(module, input, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    # Register hook to the final convolutional layer
    final_conv_layer = model.layer4[2].conv3
    hook = final_conv_layer.register_forward_hook(hook_function)

    # Forward pass
    output = model(input_tensor)
    predicted_class = output.argmax().item()

    # Backward pass
    model.zero_grad()
    output[0, predicted_class].backward()

    # Get gradients and feature maps
    gradients = gradients[0].squeeze().detach().numpy()
    feature_maps = activations[0].squeeze().detach().numpy()

    if method == "gradcam":
        # Compute Grad-CAM
        weights = np.mean(gradients, axis=(1, 2))  # Calculate the mean of the gradients
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
    elif method == "hirescam":
        # Compute HiResCAM
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i in range(feature_maps.shape[0]):
            cam += feature_maps[i] * gradients[i]
    else:
        raise ValueError("Invalid method. Use 'gradcam' or 'hirescam'.")

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    # Overlay CAM on the image
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Resize the input image to match the heatmap size
    input_image = (
        cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR) / 255
    )

    overlay = heatmap + input_image
    overlay = overlay / np.max(overlay)

    # Add predicted label to the image
    scores = F.softmax(output, dim=1).detach().numpy()
    labels = create_labels(scores, top_k=1)
    label = labels[0]
    cv2.putText(
        overlay,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Convert overlay to Image for Gradio
    overlay_image = Image.fromarray(np.uint8(255 * overlay))

    return overlay_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to the local image to be analyzed")
    parser.add_argument("--method", type=str, default="hirescam") # "gradcam" or "hirescam"
    parser.add_argument("--output", help="Name of the output image file", default="output.png")
    args = parser.parse_args()

    overlay_img = visualize_image(args.img, args.method)
    overlay_img.save(args.output)
