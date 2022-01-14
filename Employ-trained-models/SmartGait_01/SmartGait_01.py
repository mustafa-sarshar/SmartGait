# In[] Import the libraries
# disable all debugging logs for Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import pickle
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
from main.utils.preprocessing import dataset_vectorizing

# disable all debugging logs for Tensorflow
import logging
tf.get_logger().setLevel(logging.ERROR)

# In[] Initialize the primary variables and checking Command Line Arguments
if len(sys.argv) > 1:
    _gait_phase = sys.argv[1]
else:
    print("No gait phase is given!!!")
    exit(0)
_dataset_address = f"main//datasets//{_gait_phase}//db.pkl"
_model_address = f"main//models//{_gait_phase}//mobilenet"

# In[] Load the dataset
db_validation = []
with open(_dataset_address, "rb") as _data:
    db_validation = pickle.load(_data)

# In[] Initialize the constants
columns_X = [0, 1, 2]
columns_y = [3]
LSTM_window_left = 125
LSTM_window_right = 125

# In[] Initializing the train set
validation_set_X = db_validation[:, columns_X]
validation_set_y = db_validation[:, columns_y]

# In[] Validation Set
higher_bound = len(validation_set_X)-LSTM_window_right
X_validation_total, y_validation = dataset_vectorizing(validation_set_X, validation_set_y, LSTM_window_left, LSTM_window_right, higher_bound)

# In[] Load the Model
if tf.config.list_physical_devices("GPU"):
    strategy = tf.distribute.MirroredStrategy() # set to MirroredStrategy
    print("Strategy is set to MirroredStrategy")
else:
    strategy = tf.distribute.get_strategy() # set to the default strategy
    print("Strategy is set to DefaultDistributionStrategy")

with strategy.scope():
    model = models.load_model(_model_address)
    model.summary()   # show the structure of the model

    ## Prediction
    time_start = datetime.now()
    print(f"Prediction startet at: {time_start.strftime('%Y-%m-%d, %H:%M:%S')}")
    predicted_gait_phases = model.predict(X_validation_total, verbose=1)
    predicted_gait_phases = np.vstack((np.zeros((LSTM_window_left, 1)), predicted_gait_phases, np.zeros((LSTM_window_right+1, 1))))
    y_validation_corrected = np.concatenate((np.zeros((LSTM_window_left, 1)), y_validation.reshape(-1, 1),  np.zeros((LSTM_window_right+1, 1))))
    time_end = datetime.now()
    print(f"Prediction finished at: {time_end.strftime('%Y-%m-%d, %H:%M:%S')}")
    time_duration = time_end - time_start
    print(f"Duration: {time_duration}")

    ## Evaluation
    time_start = datetime.now()
    print(f"Evaluation startet at: {time_start.strftime('%Y-%m-%d, %H:%M:%S')}")
    score = model.evaluate(x=X_validation_total, y=y_validation, verbose=1)
    time_end = datetime.now()
    print(f"Evaluation finished at: {time_end.strftime('%Y-%m-%d, %H:%M:%S')}")
    time_duration = time_end - time_start
    print(f"Duration: {time_duration}")
    print(f"Evaluation results: \tAccuracy: {score[1]:0.4f}, Loss: {score[0]:0.4f}")

# In[] Visualising the results
x_timeline = list(range(len(validation_set_X[:, [0]])+1))
plt.clf()
plt.plot(validation_set_X[:, [0]], label="gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label="freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label="rotMatMag", linewidth=1, linestyle="solid", marker=".")

plt.plot(y_validation_corrected, label=f"Real {_gait_phase}", linewidth=3, color="magenta", linestyle="solid", marker="s")
markerline, stemlines, baseline = plt.stem(x_timeline, predicted_gait_phases, "r", markerfmt="ro", label=f"Predicted {_gait_phase}")
plt.setp(stemlines, 'linewidth', 3), plt.setp(markerline, 'markersize', 3)

plt.title(f"{_gait_phase} phase prediction")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Correcting the Results
peaks, properties = find_peaks(predicted_gait_phases.flatten(), height=(.1, None), distance=100)
predicted_gait_phases_corrected = np.zeros(len(predicted_gait_phases))

for index, val in enumerate(predicted_gait_phases):
    if val >= 0.05:   # set the minimum probability to equal or greater than 0.05
        predicted_gait_phases_corrected[index] = 1

# In[] Correct the predicted gait phases
for ii, ival in enumerate(predicted_gait_phases_corrected):
    if ival == 1:
        predicted_gait_phases_corrected[ii+1:ii+50] = 0

_no_of_events = len(y_validation_corrected[y_validation_corrected == 1])
_no_of_phases_predicted = len(predicted_gait_phases_corrected[predicted_gait_phases_corrected == 1])

print(f"No. of sample frames for validation dataset: {len(validation_set_X)}")
print(f"{_gait_phase} labeled: {_no_of_events}")
print(f"{_gait_phase} predicted: {_no_of_phases_predicted}")

# In[] Plot the final results
plt.clf()
plt.plot(validation_set_X[:, [0]], label="gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label="freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label="rotMatMag", linewidth=1, linestyle="solid", marker=".")

plt.plot(2*y_validation_corrected, label=f"Real {_gait_phase}", linewidth=3, color="magenta", linestyle="solid", marker="s")
plt.plot(predicted_gait_phases_corrected, label=f"Predicted {_gait_phase} (corrected)", linewidth=3, color="red", linestyle="solid", marker="s")

plt.title(f"{_gait_phase} Phase Prediction, {_no_of_phases_predicted} {_gait_phase}(s) detected!")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Finish