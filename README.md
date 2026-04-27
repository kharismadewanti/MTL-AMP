# MTL-AMP
This project implements a Multi-Task Learning (MTL) model that performs semantic segmentation and depth estimation simultaneously. Built on a ResNet50 encoder with dual decoders and skip connections, the model leverages Automatic Mixed Precision (AMP) to improve computational efficiency and reduce memory usage.

This project implements a multi-task perception model that simultaneously performs:
- Semantic Segmentation (19 classes)
- Monocular Depth Estimation

The model adopts a shared ResNet50 encoder with dual decoder branches and skip connections, enabling efficient feature sharing across tasks. To improve computational efficiency and reduce memory usage, training is performed using Automatic Mixed Precision (AMP).

This project uses a private dataset and is not publicly available.
The dataset consists of:
- RGB images
- Semantic segmentation labels
- Depth maps

Each sample must have:
- Corresponding RGB image
- Segmentation map (*_map.png)
- Depth map (.png)
