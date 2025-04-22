# CNN_Waste_Segregation
Waste Material Segregation using CNNs

This repository contains a complete solution for an AI-powered waste classification system using Convolutional Neural Networks (CNNs) implemented in PyTorch. The goal is to automate waste sorting into seven categories (Cardboard, Food Waste, Glass, Metal, Other, Paper, Plastic) to improve recycling efficiency and reduce manual sorting costs.

Repository Structure

├── data/                        # folder for the image dataset (not included)
│   ├── Cardboard/
│   ├── Food_Waste/
│   ├── Glass/
│   ├── Metal/
│   ├── Other/
│   ├── Paper/
│   └── Plastic/
├── notebooks/
│   └── Waste_Classification_PyTorch.ipynb   # solution notebook
├── models/
│   └── waste_effnet_b0.pth      # trained model weights
├── label_map.json               # Class index → label mapping
├── requirements.txt             # Python dependencies
└── README.md                    # This file

Note: The raw image data is not stored in this repository. Download the dataset separately and place it under data/ following the structure above.

Prerequisites

Python 3.11 or newer (64-bit)

PyTorch with appropriate CUDA support or CPU-only

torchvision, scikit-learn, matplotlib, seaborn, pandas

NVIDIA GPU drivers if using GPU

Install dependencies:

pip install -r requirements.txt

Example requirements.txt:

torch>=2.2.0+cu121
torchvision>=0.15.0+cu121
scikit-learn>=1.4.2
matplotlib>=3.7.1
seaborn>=0.13.2
pandas>=2.2.2

Usage

Place your dataset in data/ with subfolders for each class.

Open the notebook:

code notebooks/Waste_Classification_PyTorch.ipynb

Select the correct Python interpreter in VS Code.

Run all cells to execute end-to-end:

Data loading and visualization

Baseline CNN training

Transfer learning with EfficientNet-B0

Evaluation metrics and confusion matrix

Model saving (models/waste_effnet_b0.pth)

To run as a script:

jupyter nbconvert --to script notebooks/Waste_Classification_PyTorch.ipynb --output train.py
python train.py

Results

Baseline CNN: training loss improved over 3 epochs.

EfficientNet-B0 transfer learning: achieved over 90% validation accuracy.

Confusion matrix shows Paper vs. Cardboard had the most misclassifications.

See the notebook for detailed plots and metrics.




