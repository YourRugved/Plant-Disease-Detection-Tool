# Plant Disease Detection using YOLOv8

This repository contains a deep learning pipeline to identify 38 different types of plant diseases using the **YOLOv8** architecture. The project is designed to run seamlessly in **Google Colab** using GPU acceleration.

## 🚀 Quick Start (Google Colab)
1. Upload the `Plant_Disease_Detection.ipynb` to your Google Drive.
2. Open it with **Google Colab**.
3. Go to `Runtime` > `Change runtime type` and select **T4 GPU**.
4. Run all cells.

## 📊 Dataset
The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) via Kaggle.
- **Images:** ~70,295
- **Classes:** 38 (Apple, Grape, Tomato, Potato, etc.)
- **Diversity:** Includes both healthy and diseased leaf samples.

## 🛠️ Technical Stack
- **Framework:** Ultralytics (YOLOv8)
- **Data Management:** Kagglehub 
- **Visualization:** Matplotlib, Seaborn

## 📈 Key Workflow
1. **Automated Data Fetching:** Uses `kagglehub` to download the dataset directly into the Colab environment.
2. **EDA:** Generates distribution plots for image counts across the 38 classes to verify dataset balance.
3. **Training:** Fine-tunes the YOLOv8 Nano (`yolov8n.pt`) model for high-speed, accurate detection.
4. **Evaluation:** Provides training loss and accuracy metrics visualization.

## 📁 Repository Structure
- `Plant_Disease_Detection.ipynb`: The primary Jupyter Notebook for training and inference.
- `README.md`: Project documentation.
- . Making it "Colab-Ready"
Colab: # 1. SETUP: Install required libraries
!pip install -q kagglehub ultralytics openpyxl seaborn tqdm

# 2. DATASET: The following code downloads the dataset automatically
import kagglehub
import os

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Dataset location:", path)

# 3. DIRECTORY CONFIG: Set the training path
train_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train")
print("Train directory set to:", train_dir)

Make sure you have a cell that defines the model:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
Add a "Open in Colab" Button: You can add this HTML to the top of your README.md to make it easy to test your code:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/Plant_Disease_Detection_YOLOv8.ipynb)
