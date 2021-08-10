# SmartGait
Gait Phase Estimation by employing Artificial Intelligence

This project aims to pave the way for the development of IMU-based gait phase estimation by employing the state-of-the-art technology, in particular deep learning algorithms.

This repository consists of:
1) "Use_trained_models" directory, please check use_trained_models directory.
2) "Create_models_from_scratch" directory, for those who would like to create these models on their own from scratch.

Moreover, the project will grow itselft step-by-step, therefore more trained models will be presented in this repo in the future.
At this moment, the first SmartGait model, "SmartGait_01" is already trained by using data from only two individuals, which is still not a completed trained model for production, however, it is a proof of our concept in using LSTM algorithm to estimate three main gait phases, MidSwing, FootOff and FootContact.

## Create a virtual environment for this project as follow:
### Create a new virtual environment by choosing a Python interpreter and making a .\venv directory to hold it:
1) python -m venv .\venv
### Activate the virtual environment:
2) .\venv\Scripts\activate
### Install required libraries from the requirements.txt file
3) pip install -r requirements.txt

### Note: To exit the virtual environment later:
-) deactivate  # don't exit until you're done with your project