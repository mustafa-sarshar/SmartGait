# SmartGait
## Gait Phase Estimation by employing Artificial Intelligence

This project aims to pave the way for the development of IMU-based gait phase estimation by employing the state-of-the-art technology, in particular deep learning algorithms.

This repository consists of:
1) "Employ-trained-models" directory, please check Employ_trained_models directory if you want to use and test the trained models.
2) "Design-models-from-scratch" directory, for those who would like to design these models on their own from scratch.

Moreover, the project will grow itselft step-by-step, therefore more trained models will be presented in this repo in the future.
At this moment, the first SmartGait model, "SmartGait_01" is already trained by using data from **only** two individuals, which is still not a completed trained model for production, however, it is a proof of our concept for using LSTM algorithm to estimate three main gait phases, foot-off, mid-swing, and foot-contact.

### Install Python
*) For this Repo the Python version 3.9.6 was used. Please download Python from: https://www.python.org/downloads/
### Create a virtual environment for this project as follows:
#### Create a new virtual environment by choosing a Python interpreter and making a .\venv directory to hold it:
1) python -m venv .\venv
#### Activate the virtual environment:
2) .\venv\Scripts\activate
#### Install required libraries from the requirements.txt file
3) pip install -r requirements.txt

#### Note: To exit the virtual environment later:
-) deactivate  # don't exit until you're done with your project

### Related Articles/Publications:
*) Gait Phase Estimation by Using LSTM in IMU-Based Gait Analysis—Proof of Concept [Link: https://www.mdpi.com/1424-8220/21/17/5749]
