import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sys
import yaml

# Load configuration
with open("./CONFIG.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def main(image_path, model_weights):

    # Define class names for cats and dogs
    class_names = ['cat', 'dog']

    # Load pre-trained model and set to evaluation mode
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))

    # Load the checkpoint
    checkpoint = torch.load(model_weights, map_location=torch.device('cpu'))

    # Extract the state_dict
    state_dict = checkpoint['state_dict']

    # Remove the 'model.' prefix from state_dict keys if present
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    # Load the state dict into the model
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

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

    # Compute Grad-CAM
    weights = np.mean(gradients, axis=(1, 2))  # Calculate the mean of the gradients
    grad_cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        grad_cam += w * feature_maps[i]

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (224, 224))
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam = grad_cam / np.max(grad_cam)
    grad_cam = np.uint8(255 * grad_cam)  # Convert to uint8

    print("Shape of grad_cam:", grad_cam.shape)  # Debugging line

    # Overlay Grad-CAM on the image
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Resize the input image to match the heatmap size
    input_image = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR) / 255

    overlay = heatmap + input_image
    overlay = overlay / np.max(overlay)

    # Add predicted label to the image
    label = class_names[predicted_class]
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save and display result
    cv2.imwrite('grad_cam_output.jpg', np.uint8(255 * overlay))

if __name__ == "__main__":
    file = sys.argv[1]
    main(file, cfg["MODEL_WEIGHTS"])
