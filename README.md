# Image-Caption-Generator

## Description

This project implements an image captioning model using the COCO dataset. The code extracts image features using the pre-trained VGG16 model and then uses a GRU-based decoder to generate captions for the images.

### Key Features:
- **VGG16 Model**: Used for feature extraction from images.
- **GRU Decoder**: Used for caption generation.
- **COCO Dataset**: Utilized for training the model.
- **Tokenizer**: For processing captions and converting them into sequences.
- **Model Checkpoints**: Saving the model weights at different checkpoints.
- **TensorBoard**: For monitoring the training process.

## Installation

To run the project, you need to install the following dependencies:

```bash
pip install tensorflow numpy matplotlib pillow