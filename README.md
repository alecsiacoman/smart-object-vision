# Smart Object Recognition: Detection of Cups

## Introduction

### Problem Definition
This project aims to build an **object detection system** capable of identifying cups. The model will be trained using a dataset of labeled images and deployed in a real-world scenario through a **mobile application** that captures photos and identifies the object.

This task falls under **pattern recognition** and **computer vision**, with practical relevance in retail inventory management, smart kitchen assistants, waste sorting, and accessibility tools.

### Importance & Context
In everyday environments, being able to automatically recognize simple household items like mugs or bottles is a foundational step toward **intelligent vision systems**.  
Applications include:
- **Retail automation:** automatic product recognition at checkout.
- **Smart home devices:** identifying utensils or containers.
- **Assistive technology:** describing items to users via voice.
- **Mobile AR experiences:** recognizing physical items through a phone camera.
  

## Datasets

We use a **subset of the COCO dataset (Common Objects in Context)** — one of the most comprehensive and widely used datasets for object detection.

COCO provides over 118,000 training images and 5,000 validation images, each with **bounding box annotations** for 80 object categories.  
We will extract only the relevant category for our task:
- **cup** (category ID: 47)

This subset gives us a large, diverse, and realistic dataset that already contains high-quality annotations of cups in complex, real-world environments (e.g., kitchens, desks, restaurants).
Available at [https://cocodataset.org](https://cocodataset.org)

### Dataset Construction  
To build a meaningful training and evaluation setup, we perform the following steps:

1. **Extract only images containing cups** from COCO’s train and validation splits.  
2. **Convert bounding box annotations** to YOLO-compatible format for YOLOv8, and retain COCO JSON format for Faster R-CNN.  
3. **Add negative samples**:
   - A subset of images **without cups** is intentionally included in both the **train** and **test** splits.
   - This ensures that both models are evaluated not only on how well they detect cups, but also on **their ability to correctly identify when no cup is present** (i.e., avoid false positives).  

The final dataset therefore contains:
- **Positive samples** (cup images)  
- **Negative samples** (images with no cups)

This setup better reflects realistic conditions and supports a fair comparison between the two detection models.

## Models Compared

We compare two widely used object detection models with different architectural approaches:

### 1. YOLOv8 (You Only Look Once – Version 8)
- **One-stage detector**
- Performs object localization and classification in a single forward pass
- Optimized for speed and real-time inference
- Implemented using the **Ultralytics YOLOv8** framework

### 2. Faster R-CNN
- **Two-stage detector**
- First stage generates region proposals using a Region Proposal Network (RPN)
- Second stage classifies and refines bounding boxes
- Known for strong accuracy and robustness on smaller datasets
- Implemented using **Torchvision**

This comparison highlights the trade-offs between **speed-oriented** and **accuracy-oriented** detection models.

## Evaluation Metrics

Model performance is evaluated using standard **object detection metrics**, consistent with the COCO evaluation protocol:

- **Intersection over Union (IoU)**  
  - IoU threshold: **0.5**
- **Average Precision (AP)**  
  - Measures detection accuracy across confidence thresholds
- **Average Recall (AR)**  
  - Measures the model’s ability to detect all relevant objects


## Experimental Setup

- Same training, validation, and test splits for both models
- Same object class (**cup**)
- Same evaluation metrics
- Pre-trained weights used for model initialization
- Training and evaluation performed using PyTorch-based frameworks

