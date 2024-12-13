# Nuclei Segmentation in H&E-Stained Histological Images using Detectron2 

<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<!-- Issue & Pull Request Badges -->
<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<!-- Repository Activity & Stats Badges -->
<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<!-- Language & Code Style Badges -->
<div align="center">
    <img src="https://img.shields.io/github/languages/count/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count">
    <img src="https://img.shields.io/github/languages/top/arpsn123/31ShadesofHE?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<!-- Maintenance Status Badge -->
<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>



## Overview

This project uses Detectron2 for nuclei segmentation in Hematoxylin and Eosin (H&E)-stained histological images from 31 distinct human and mouse organs. Our dataset consists of **665 image patches**, containing **over 30,000 manually segmented nuclei** with annotations converted to COCO JSON format compatible with Detectron2. The project outputs segmented masks and visualizations to aid in biomedical analysis.

## Dataset Details

The nuclei were fully manually annotated, with 665 image patches derived from 31 organs to represent diverse tissues and structures. Detailed sub-sections for each organ group highlight their role in the dataset:

### Human Organs
This subset includes 21 organs, each carefully chosen to represent a comprehensive range of human histological structures and their unique cellular features:

- **Bladder**: As a hollow organ, bladder samples offer insights into the nuclei structure of urothelial cells.
- **Brain**: Brain samples contribute a unique type of nuclei from neurons, essential for analyzing complex brain tissue.
- **Cardia**: Nuclei data from cardia sections offer insights into the gastroesophageal junction.
- **Cerebellum**: Specialized neurons and glial cells make cerebellum samples a valuable source of varied nuclei.
- **Epiglottis**: Highlights nuclei structures involved in respiratory and digestive pathways.
- **Jejunum**: A part of the small intestine, jejunum samples showcase epithelial and villous cell nuclei.
- **Kidney**: Renal tissues provide nuclei patterns critical to kidney function studies.
- **Liver**: Liver nuclei reflect hepatic cells and are integral for analyzing metabolic tissue structure.
- **Lung**: Pulmonary tissue nuclei illustrate respiratory cellular structures.
- **Melanoma**: Includes nuclei of cancerous cells, enabling research into malignant growth patterns.
- **Muscle**: Muscle fibers present in these samples show nuclei in a highly organized structure.
- **Esophagus**: Nuclei from esophageal tissue assist in understanding mucosal layers and cell turnover.
- **Pancreas**: Pancreatic nuclei help study exocrine and endocrine function cell types.
- **Peritoneum**: Peritoneal samples capture nuclei from serous membranes, valuable for epithelial tissue analysis.
- **Placenta**: Placental cells add nuclei essential for reproductive and developmental studies.
- **Pylorus**: The pyloric part of the stomach, where nuclei patterns relate to digestive functions.
- **Rectum**: Rectal samples help in analyzing nuclei involved in lower digestive tract tissue.
- **Salivary Gland**: Offers nuclei from glandular cells with a secretory role.
- **Spleen**: Spleen tissue adds nuclei associated with immune function.
- **Testis**: Testicular nuclei from seminiferous tubules support reproductive studies.
- **Tonsil**: Lymphoid tissue nuclei from tonsils aid in immune response analysis.
- **Umbilical Cord**: Embryonic tissue nuclei provide data relevant to development and stem cell research.

### Mouse Organs
This subset includes nuclei data from 10 organs, adding diversity in cellular structure from murine samples to complement human data:

- **Subscapular (White/Brown)**: Contains fat tissue from different body regions, revealing adipocyte nuclei.
- **Femur**: Bone marrow samples give nuclei involved in hematopoiesis.
- **Heart**: Cardiac muscle nuclei from the murine heart support studies on cardiac health.
- **Kidney**: Rodent renal tissue nuclei provide insights into kidney physiology.
- **Liver**: Similar to human samples, hepatic cells from mice support liver-related studies.
- **Tibia Muscle**: Limb muscle nuclei illustrate structural organization in murine muscle tissue.
- **Spleen**: Murine spleen nuclei are used to study immune responses.
- **Thymus**: This lymphoid organ in mice provides nuclei for immune system development analysis.

## Data Preparation

The dataset used in this project consists of **665 image patches** that represent various histological samples of human and mouse organs, each containing manually annotated nuclei. The preparation process for this dataset involved several key steps to ensure that the images and annotations were compatible with Detectron2 and ready for model training:

### 1. Image Preprocessing
The raw histological images, stained with **Hematoxylin and Eosin (H&E)**, were initially cleaned and preprocessed to ensure clarity and high-quality segmentation. This involved:
- **Resizing**: Standardizing the image sizes to fit the model input requirements.
- **Normalization**: Adjusting image pixel values to normalize the intensities across different slides, ensuring consistency in staining and lighting variations.
- **Augmentation**: Data augmentation techniques like rotation, flipping, and scaling were applied to increase the diversity of the training dataset and improve model robustness.

### 2. Annotation Conversion
The manual annotations for each image patch were originally provided in a custom format and needed to be converted to **COCO JSON format** to be compatible with Detectron2. This step involved:
- **Manual Segmentation**: Every nucleus in the image was manually marked and labeled. Each annotation was assigned a unique identifier for each instance of a nucleus, with the background labeled as 0 (black) to differentiate it from the foreground (nuclei).
- **Bounding Boxes and Masks**: Each nucleus in the image was represented by both a **bounding box** (for object detection) and a **segmentation mask** (for instance segmentation), stored in the converted COCO JSON file. This file contains all the necessary information, such as:
  - Image dimensions
  - Annotations including category (nucleus or background), polygon points, and bounding box coordinates.
  - Object segmentation in the form of binary masks

### 3. Labeling and Class Distribution
Each image patch contains **over 30,000 segmented nuclei**, which were all labeled under a single class for nuclei segmentation. The class labels were organized into a consistent format, ensuring compatibility with Detectron2’s instance segmentation capabilities. The distribution of nuclei within the dataset varies across the different organs, making it essential for the model to learn generalizable features.

### 4. COCO JSON Structure
The COCO JSON files were structured to contain the following essential elements:
- **Images**: Metadata about each image patch, such as filename, image size (height and width), and ID.
- **Annotations**: A detailed list of all annotated nuclei, including their segmentation masks (as polygons) and bounding boxes.
- **Categories**: The class labels for the annotations, in this case, only one category for nuclei.
- **Additional Fields**: Any extra metadata such as image IDs, segmentation polygons, and confidence scores were also included to enhance model training and validation,

## Tech Stack

### Frameworks and Libraries
- ![Detectron2](https://img.shields.io/badge/Detectron2-2.0.0-blue.svg?style=for-the-badge&logo=detectron2)
- ![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-ff7f0e.svg?style=for-the-badge&logo=pytorch)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-5C3A3A.svg?style=for-the-badge&logo=openCV)
- ![NumPy](https://img.shields.io/badge/NumPy-1.24.2-blue.svg?style=for-the-badge&logo=numpy)
- ![matplotlib](https://img.shields.io/badge/matplotlib-3.6.0-blue.svg?style=for-the-badge&logo=matplotlib)

### Dataset and Formats
- ![COCO API](https://img.shields.io/badge/COCO--API-v2.0-green.svg?style=for-the-badge&logo=coco)
- ![COCO Format](https://img.shields.io/badge/COCO--Format-v2.0-yellow.svg?style=for-the-badge&logo=coco)

### Tools and Platforms
- ![Google Colab](https://img.shields.io/badge/Google_Colab-free-lightgreen.svg?style=for-the-badge&logo=googlecolab)
- ![Google Drive](https://img.shields.io/badge/Google_Drive-15GB_Free-lightblue.svg?style=for-the-badge&logo=googledrive)
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?style=for-the-badge&logo=jupyter)

### Development and Version Control
- ![Git](https://img.shields.io/badge/Git-2.47.0-orange.svg?style=for-the-badge&logo=git)
- ![GitHub](https://img.shields.io/badge/GitHub-arpsn123-success.svg?style=for-the-badge&logo=github)

### Others
- ![TensorBoard](https://img.shields.io/badge/TensorBoard-2.10.0-orange.svg?style=for-the-badge&logo=tensorboard)
- ![Shapely](https://img.shields.io/badge/Shapely-1.8.1-green.svg?style=for-the-badge&logo=shapely)
- ![TorchServe](https://img.shields.io/badge/TorchServe-0.5.0-lightblue.svg?style=for-the-badge&logo=torchserve)
- ![NVIDIA](https://img.shields.io/badge/NVIDIA-GTX_1080-green.svg?style=for-the-badge&logo=nvidia)
- ![Numba](https://img.shields.io/badge/Numba-0.56.0-red.svg?style=for-the-badge&logo=numba)


## Model & Training

### 1. **Model Configuration and Setup**
   The training process begins with configuring the Detectron2 model for instance segmentation, focusing on nuclei detection in histological images. The following steps were taken to initialize the configuration:

   - **Model Configuration**: The configuration file for the Mask R-CNN model with a ResNet50 backbone and Feature Pyramid Network (FPN) was loaded from the Detectron2 model zoo. This model was pre-trained on the COCO dataset, providing a solid starting point for training on the nuclei dataset.
     ```python
     from detectron2.engine import DefaultTrainer
     from detectron2.config import get_cfg
     from detectron2 import model_zoo
     import os

     cfg = get_cfg()
     cfg.OUTPUT_DIR = "/content/drive/MyDrive/Nuclei_Segmentation_&_Analysis_using_Detectron2_&_YOLOv8/model_store"
     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
     ```

   - **Dataset Specification**: The training and testing datasets were set up. The dataset was registered using Detectron2’s Dataset Catalog API, with the training dataset being specified while leaving the test dataset empty for evaluation.
     ```python
     cfg.DATASETS.TRAIN = ("my_dataset_train",)
     cfg.DATASETS.TEST = ()
     ```

### 2. **Training Configuration**
   Several key parameters were tuned to improve the training process:

   - **Number of Workers**: This parameter controls how many CPU threads are used to load the data during training. Two workers were allocated for efficient data loading.
     ```python
     cfg.DATALOADER.NUM_WORKERS = 2
     ```

   - **Pretrained Weights**: The model was initialized with weights pre-trained on the COCO dataset to leverage the learned features from a large, diverse dataset. The weights for the Mask R-CNN model were pulled from the model zoo.
     ```python
     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
     ```

   - **Batch Size and Learning Rate**: The training batch size was set to 2 images per batch, and the learning rate was set to 0.00025 to allow for stable training.
     ```python
     cfg.SOLVER.IMS_PER_BATCH = 2
     cfg.SOLVER.BASE_LR = 0.00025
     ```

   - **Iterations and Training Steps**: The model was set to train for 1500 iterations, with no specific learning rate schedule (steps).
     ```python
     cfg.SOLVER.MAX_ITER = 1500
     cfg.SOLVER.STEPS = []
     ```

   - **Model-specific Settings**: The Region of Interest (ROI) heads were adjusted to process 256 samples per image. The number of classes was set to 6, considering the number of object categories (e.g., different tissue types or nuclei).
     ```python
     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
     ```

### 3. **Training Execution**
   After configuring the model, the output directory for storing the trained models was created, ensuring that all checkpoints and results were saved during training:
   ```python
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
```
The model was then trained using the DefaultTrainer class from Detectron2, which manages the entire training loop and evaluation process. The model began from scratch, as specified by resume=False.

```
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
```

The training loop ran for 1500 iterations, with the loss and evaluation metrics being output at regular intervals. The following is a sample of the output during training:

```
05/02 18:02:19 d2.utils.events]:  eta: 0:00:00  iter: 1499  total_loss: 1.403  loss_cls: 0.2948 
```

## Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.8+
- pip (Python package installer)
- CUDA-enabled GPU (for faster training)

## Installation Steps

### 1. Clone the repository



```bash
git clone https://github.com/arpsn123/31ShadesofHE.git
cd 31ShadesofHE
```
### 2. Set up a Python environment
Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```
### 3. Install dependencies
Install the required libraries:

```pip install -r requirements.txt```
### 4. Install Detectron2
Follow the official Detectron2 installation instructions based on your CUDA version. Example for CUDA 11.1:
```
pip install torch==1.8.1 torchvision==0.9.1
pip install detectron2==0.6
```

### 5. Prepare the Dataset
Place your dataset in the correct folder and follow the structure defined by the project.

Convert annotations into COCO format if they aren't already.

Update the ```my_dataset_train``` dataset path in the configuration file to point to your dataset.

### 6. Configure Training
In the ```train.py``` script, make sure the paths and settings are properly configured:

Data paths: Ensure the dataset paths are correct.
Output directory: Set the output directory for model checkpoints.
Hyperparameters: Adjust parameters like learning rate, batch size, and iterations as needed.

### 7. Start Training
Run the training script:

```
python train.py
```
The model will train and save results to the specified output directory.

## Evaluation
After training, you can evaluate the model using the saved checkpoints. The evaluation metrics will be outputted in a table with AP, AP50, and other related scores.


## Result & Analysis

### 1. **Output Folders and Files**
   
After training the model, the following output files and folders were generated to store the training results, evaluation metrics, and the final trained model:

- **model_store Folder**: This directory contains critical files related to the training process, including:
  
  - **Config.yaml**: This file contains the configuration used during training. It stores all hyperparameters, model settings, and dataset information. This file ensures that the training setup can be reproduced, and it provides insight into the configuration choices made during the training process.
  
  - **events.out.tfevents.1714672154.7de69b4c3373.2008.0**: This is a TensorFlow events file, used for tracking various metrics, such as training loss and evaluation metrics, over the course of the training. These logs can be visualized in TensorBoard, providing a clear and interactive view of the model's performance during training.
  
  - **last_checkpoint**: This file stores the checkpoint of the model at the most recent iteration. It allows for resuming training from the last saved state if necessary. If training is interrupted or needs to be continued, this checkpoint serves as the starting point for further training.
  
  - **metrics.json**: This file contains the evaluation metrics for the model after training. It includes performance indicators such as **Average Precision (AP)**, **IoU (Intersection over Union)**, and other evaluation metrics that give insight into the model's performance across different object categories and tasks.
  
  - **model_final.pth**: This is the final saved model after all the training iterations. It contains the weights learned by the model and can be used for inference or further fine-tuning. The final model represents the culmination of the entire training process and is the most reliable model for making predictions.

- **Additional Folders**:
  
  - **predicted-ground-truth**: This folder contains grayscale images where each detected nucleus is marked with intensity values between 1 and 255. These images serve as a direct comparison to the ground-truth data for evaluating model performance.
  
  - **detectron-segmented**: This folder contains images with segmented nuclei, where bounding boxes and colored overlays are applied to represent detected objects. These images provide a visual assessment of how well the model has segmented and localized the nuclei.

### 2. **Evaluation Results**
   
The **metrics.json** file contains the evaluation results of the model on the test data. The following are the key metrics that were recorded for both bounding box and segmentation tasks:

- **Bounding Box (bbox)**:

  These values reflect the model's performance in detecting and localizing objects (nuclei) in the images.
  
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
  |:------:|:------:|:------:|:------:|:------:|:-----:|
  | 34.396 | 65.189 | 33.320 | 30.536 | 45.138 |  nan  |
  
  - **AP** (Average Precision): 34.40 — This indicates the model's overall ability to accurately detect the objects across different IoU thresholds.
  
  - **AP50** (AP at IoU threshold of 50%): 65.19 — The model performs particularly well at a 50% IoU threshold, suggesting that it is good at detecting objects with significant overlap.
  
  - **AP75** (AP at IoU threshold of 75%): 33.32 — The model's performance drops when the overlap requirement is stricter, but it still maintains a moderate level of accuracy.

  - **APs**, **APm**, **APl**: These represent AP values for small, medium, and large objects. Here, **APm** (for medium-sized objects) is 45.14, suggesting that the model performs better with medium-sized nuclei compared to smaller ones.

- **Segmentation (segm)**:

  These values reflect the model’s performance in pixel-wise segmentation of the nuclei.

  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
  |:------:|:------:|:------:|:------:|:------:|:-----:|
  | 31.334 | 62.359 | 28.575 | 25.574 | 44.969 |  nan  |
  
  - **AP**: 31.33 — The model's performance in segmentation is slightly lower than in detection but still significant. This reflects the model's ability to predict the exact boundaries of the nuclei.
  
  - **AP50**: 62.36 — Again, the model performs well when a more relaxed overlap criterion (50%) is used.
  
  - **AP75**: 28.58 — Similar to bounding box results, the segmentation performance drops with higher IoU thresholds.
  
  - **APs**, **APm**, **APl**: The model's segmentation accuracy is better for medium-sized nuclei (**APm**: 44.97), while the performance for smaller nuclei (**APs**: 25.57) is less accurate.

### 3. **Insights from the Results**
   
The evaluation results provide valuable insights into the model's strengths and areas for improvement:

- The model achieves relatively high **AP50** scores for both bounding box and segmentation tasks, suggesting that it is quite capable of detecting and segmenting the nuclei when the overlap requirement is not too strict.
  
- The **APs** values (for small objects) indicate that the model struggles with smaller nuclei, which could be due to their less distinct features or insufficient representation in the training data. Improving the model's ability to handle smaller objects could involve adjustments such as augmenting the dataset or tuning hyperparameters to improve sensitivity to small objects.

- The **APm** values for medium-sized objects are the highest in both detection and segmentation, indicating that the model excels in these categories. This could suggest that nuclei of medium size have more distinct features, making them easier to detect and segment accurately.

- The performance for larger nuclei (**APl**) could not be evaluated (nan), which suggests that the dataset may not have had enough large nuclei samples to draw reliable conclusions.

### 4. **Conclusion**

The model's training results indicate that the **Mask R-CNN** architecture, pre-trained on COCO, is well-suited for nuclei segmentation in histological images. While the model performs best with medium-sized nuclei, further work is needed to improve its detection and segmentation of small and large nuclei. Future iterations of the model may benefit from additional training data, further fine-tuning, and possibly using a more specialized backbone to enhance performance across all object sizes.



