Medical Image Analysis: Classification & Segmentation 🩺🧠
This project focuses on applying Deep Learning methods to analyze medical images using PyTorch. 
The project is divided into two main parts: binary classification of 2D chest X-rays and 3D segmentation of the hippocampus in MRI scans.

Special attention is given to the specifics of medical data: severe class imbalance, medical evaluation metrics (Sensitivity, Specificity, Dice Score), and Explainable AI (XAI) techniques.
🗂 Project Structure
Part 1: 
Pneumonia Classification on X-rays (2D)Dataset: Pneumonia 
MNIST (64x64 pixel chest X-rays).
Task: Binary classification (Normal / Pneumonia).
Architecture: Custom Convolutional Neural Network (CNN).
Key Features:
Data augmentation via random affine transformations.
Decision threshold tuning to achieve a medical-grade Sensitivity (Recall) of $\ge 0.95$.Explainable AI (XAI): 
Implementation of Grad-CAM and Occlusion Sensitivity to visualize the regions (heatmaps) the neural network focuses on when making a diagnosis.

Part 2: Hippocampus Segmentation on MRI 
(3D)Dataset:  Medical Segmentation Decathlon (Task04_Hippocampus).
Task: Segmentation of the anterior and posterior parts of the hippocampus.
Architecture: 2D U-Net (processing 3D volumes as a series of 2D axial slices with zero-padding).
Key Features:
MRI-specific normalization: intensity clipping (0.5th - 99.5th percentiles) and Z-score normalization.
Loss function: A combination of CrossEntropy and Soft Dice Loss to handle the heavily dominant background class.
Post-processing: Extracting the Largest Connected Component to eliminate noise.
Volumetry (Quantitative Biomarkers): Converting predicted pixels into physical volume ($mm^3$) and evaluating performance using Mean Absolute Percentage Error (MAPE) and Pearson correlation. Further evaluation using 3D Bounding Boxes and 3D Intersection over Union (IoU).🚀

Tech StackPython 3.xPyTorch / Torchvision (Deep Learning framework) NumPy / SciPy (Matrix computations and image processing algorithms) NiBabel (Reading and writing medical neuroimaging formats like .nii.gz)Scikit-learn (ROC-AUC calculation, Confusion Matrices)Matplotlib (Plotting and XAI visualization)

Installation & UsageClone the repository:Bashgit clone https://github.com/skrilaq/Practicum-on-ML-AI-on-visual-data-Medical-image-analysis.git

cd /Practicum-on-ML-AI-on-visual-data-Medical-image-analysis

Install the required dependencies:Bashpip 

install torch torchvision numpy scipy nibabel scikit-learn matplotlib

Launch Jupyter Notebook / Google Colab:
Bash jupyter notebook

Note: Datasets are downloaded and extracted automatically when you run the first code cell.

This project was developed as part of a Medical Image Analysis course.
