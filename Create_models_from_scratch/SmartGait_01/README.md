# SmartGait_01
## The model has been described in the article below:
### Title: [Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis — Proof of Concept](https://www.mdpi.com/1424-8220/21/17/5749)

### Please cite this article if you want to you use or employ these models in your projects as follows:
#### MDPI and ACS Style
Sarshar, M.; Polturi, S.; Schega, L. Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept. Sensors 2021, 21, 5749. https://doi.org/10.3390/s21175749

#### AMA Style
Sarshar M, Polturi S, Schega L. Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept. Sensors. 2021; 21(17):5749. https://doi.org/10.3390/s21175749

#### Chicago/Turabian Style
Sarshar, Mustafa, Sasanka Polturi, and Lutz Schega. 2021. "Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept" Sensors 21, no. 17: 5749. https://doi.org/10.3390/s21175749

## About this directory:
### This directory contains the source codes to build the SmartGait_01 model from scratch.

1) 01_DataExtraction_DatasetInitialization.py
- Data extraction & dataset initialization
   - Data extraction & dataset initialization
   - Extract the sub dataset 
   - Extract necessary features
   - Save it as pickle

2) 02_Data_Preprocessing.py
- Data preprocessing:
   - Missing value imputation
   - Remove the outliers
   - Smooth and filter the data
   - Add new features to the dataset
   - Cut the dataset and reduce its size if necessary

3)1) 03_DataLabeling_testdataset.py
- Data labeling for Test set:
    - HeelStrike
    - ToeOff
    - MidSwing

3.2) 03_DataLabeling_traindataset.py
 -   Data labeling for Train set:
    - HeelStrike
    - ToeOff
    - MidSwing

4) 04_TrainAndTest.py
Train and Test the model
    - Initialize the train and test datasets
    - Vectorize the train and test datasets
    - Define and compile the model
    - Fit the model