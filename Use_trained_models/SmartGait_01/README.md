# SmartGait_01
## The model has been described in the article below:
### Title: Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis — Proof of Concept [a link](https://www.mdpi.com/1424-8220/21/17/5749)

### Please cite this article if you want to you use or employ these models in your projects as follows:
#### MDPI and ACS Style
Sarshar, M.; Polturi, S.; Schega, L. Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept. Sensors 2021, 21, 5749. https://doi.org/10.3390/s21175749

#### AMA Style
Sarshar M, Polturi S, Schega L. Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept. Sensors. 2021; 21(17):5749. https://doi.org/10.3390/s21175749

#### Chicago/Turabian Style
Sarshar, Mustafa, Sasanka Polturi, and Lutz Schega. 2021. "Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept" Sensors 21, no. 17: 5749. https://doi.org/10.3390/s21175749

## About this directory:
### This directory contains all models trained for SmartGait_01 and sample Python codes to run and evaluate each model.

1) SmartGait_01.py: please run this file and define the gait phase as a command line argument, if you want to separately test and evaluate each model.
- run: python SmartGait_01.py [enter the gait phase name]
- example: python SmartGait_01.py MidSwing


2) SmartGait_01_allModels.py: please run this file, if you wish to run all the models and compare their performance together.
- run: python SmartGait_01_allModels.py