# Airbus_ship_detection

## Overview
This repo includes data analisis and code to building a model for Kaggle Airbus Ship Detection Challenge
link: https://www.kaggle.com/competitions/airbus-ship-detection/overview

Main files:
 * `train.py` - trains the model
 * `ship_detection_streamlit_app.py` - creating streamlit application where the model can be tested
 * `requirements.txt` - required python modules
 * `exploratory_data_analysis.ipynb` - exploratory data analysis
 * `seg_model.h5` - the model
 * `fullres_model.h5` - the model ready to work with full-scaled images

## Description of solution
###Data exploration
In this competition, we needed to locate ships in images. More than half images do not contain ships, and those that do may contain multiple ships.
File train_ship_segmentations_v2.csv contains id and encoded pixels with places of ships. So we need to decode these pixels into mask the same size as our images.
We can reduce the images size to facilitate the learning process for the neural network. But the ships on images might be very small, so we can reduce images size only a little.
Also, because the dataset isn't balanced, for better model performance, we created better balanced train and validation datasets, by undersampling the empty images.

### Model
We used a model with U-Net architecture that is a good choice for the segmentation task.
We also used Dropout layers and data augmentation, to avoid overfitting.
Trained the model using GPU.

### Loss function and metric
We used the Dice loss function as a main metric. It's calculated as 1-dice_score. We used Dice loss instead of Dise score because we need to decrease the loss function when the model shows better results.
During the learning model achieved 0.6176 Dice loss on training dataset, and 0.6452 on test dataset.

### Deploying
To deploy the project on your local machine after clone repository, you need to install libraries from requirements.txt, and the rin next command from console:
`streamlit run ship_detection_streamlit_app.py`



