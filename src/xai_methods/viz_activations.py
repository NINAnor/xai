import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def initModel():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def processImg(image_path):

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    return input_tensor

def get_activations(model, input_tensor, layer_names):

    # Register hooks to capture activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name in layer_names:
        layer = dict(model.named_modules())[name]
        layer.register_forward_hook(get_activation(name))

    # Forward pass
    _ = model(input_tensor)

    return activations

def visualize_activation(activation, title):
    num_channels = activation.shape[1]
    size = int(np.ceil(np.sqrt(num_channels)))

    fig, axes = plt.subplots(size, size, figsize=(12, 12))
    fig.suptitle(title)

    for i in range(size * size):
        ax = axes[i // size, i % size]
        if i < num_channels:
            ax.imshow(activation[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')

    plt.show()

def main(image_path):

    # Choose layers to visualize
    layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]

    model = initModel
    input_tensor = processImg(image_path)

    activations = get_activations(model, input_tensor, layer_names)

    for name in layer_names:
        visualize_activation(activations[name], f"Activations from {name}")

if __name__ == "__main__":

    image_path = sys.argv[1]
    main(image_path)
