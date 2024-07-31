# Explainable AI visualisation

All the XAI techniques in this repository are using a pretrained Residual Network 50 layers from [torchvision](https://pytorch.org/vision/stable/index.html).

## How to use the folder

- Install the dependancies and enter the virtual environment:

```
poetry install
poetry shell
```

## User interface

We made a user interface available so you can seemlessly explain the pictures:

```
python gradio_app.py
```

## Supported methods

So far we have few supported XAI methods.

- **Grad-CAM and HiResCAM**:

The Grad-CAM technique consists of highlighting the pixels in the image that are important for the model to make its classification decision [1]. HiResCAM is a generalization of CAM that does not have the same drawbacks as Grad-CAM [2].

```bash
python src/xai_methods/cam.py --img ./assets/images/dog.png --output src/xai_methods/dog_hirescam.png --method hirescam
```

This will give you the following result:

![Picture of a German shepherd classified by an AI model (labelled as dog) and explained using Grad-CAM technique. The HiResCAM technique consists of highlighting the pixels in the image that are important for the model to make its classification decision.](./src/xai_methods/dog_hirescam.png)


- **Deep Feature Factorization**:

DFF is a technique that decomposes the image as a set of interpretable components as viewed by the AI algorithm. This allows the visualization and understanding of complex representations learned by the network [3]. 
 
```bash
python src/xai_methods/dff.py --img ./assets/images/dog.png --output src/xai_methods/dog_dff.png --n_components 2  
```

This will give you the following result:

![Picture of a German shepherd explained using Deep Feature Factorization (DFF).](./src/xai_methods/dog_dff.png)

## Reference:

[1] [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
[2] [Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks](https://arxiv.org/abs/2011.08891)
[3] [Deep Feature Factorization For Concept Discovery](https://arxiv.org/abs/1806.10206)