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
Also, because of the dataset isn't balanced, for better model performance, we created a balanced train and validation datasets.

###Model
We used a model with U-Net architecture that is a good choice for the segmentation task.
We also used Dropout layers and data augmentation, to avoid overfitting.
Trained the model using GPU.

###Loss function and metric
We used the Dice loss function as a main metric. It's calculated as 1-dice_score. We used Dice loss instead of Dise score because we need to decrease the loss function when the model shows better results.


