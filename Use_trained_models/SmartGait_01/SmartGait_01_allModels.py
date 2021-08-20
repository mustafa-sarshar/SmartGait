# In[] Import the libraries

# disable all debugging logs for Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
from main.utils import preprocessing as preProc

# disable all debugging logs for Tensorflow
import logging
tf.get_logger().setLevel(logging.ERROR)

# In[] Initialize the primary variables
_gait_phases = "FootOff, MidSwing & Footcontact"   # change the value of _gait_phase variable to either FootOff or FootContact to load and run dataset and model, for FootOff or FootContact, respectively.
_dataset_address_all = "main//datasets//All/db_all.pkl" # define the address of the dataset
_models_addresses = [
    ("FootOff", "main//models//FootOff//mobilenet"), # define the address for FootOff model
    ("MidSwing", "main//models//MidSwing//mobilenet"), # define the address for MidSwing model
    ("FootContact", "main//models//FootContact//mobilenet") # define the address for FootContact model
]

# In[] Load the dataset
db_all = []
with open(_dataset_address_all, "rb") as _data:
    db_all = pickle.load(_data)

# In[] Initialize the constants
columns_X = [0, 1, 2]
columns_y_FO = [3]
columns_y_MidS = [4]
columns_y_FC = [5]
LSTM_window_left = 125
LSTM_window_right = 125

# In[] Initializing the train sets
validation_set_X = db_all[:, columns_X]
validation_set_y = []
validation_set_y.append(db_all[:, columns_y_FO])
validation_set_y.append(db_all[:, columns_y_MidS])
validation_set_y.append(db_all[:, columns_y_FC])

# In[] Validation Sets
higher_bound = len(validation_set_X)-LSTM_window_right
y_validation = []
X_validation_total, _fo  = preProc.dataset_vectorizing(validation_set_X, validation_set_y[0], LSTM_window_left, LSTM_window_right, higher_bound)
_,                  _mids = preProc.dataset_vectorizing(validation_set_X, validation_set_y[1], LSTM_window_left, LSTM_window_right, higher_bound)
_,                  _fc = preProc.dataset_vectorizing(validation_set_X, validation_set_y[2], LSTM_window_left, LSTM_window_right, higher_bound)
y_validation.append(_fo), y_validation.append(_mids), y_validation.append(_fc)
del _fo, _mids, _fc

# In[] Load the Model
if tf.config.list_physical_devices("GPU"):
    strategy = tf.distribute.MirroredStrategy() # set to MirroredStrategy
    print("Strategy is set to MirroredStrategy")
else:
    strategy = tf.distribute.get_strategy() # set to the default strategy
    print("Strategy is set to DefaultDistributionStrategy")

with strategy.scope():
    
    predicted_gait_phases = []
    y_validation_corrected = []
    scores = []

    for mdl_index, mdl_obj in enumerate(_models_addresses):
        
        # Load each model
        print(f"Loading {mdl_obj[0]} Regression Model from:\t{mdl_obj[1]}")
        model = models.load_model(mdl_obj[1])
        model.summary() # show the structure of the model
    
        ## Prediction
        time_start = datetime.now()
        print(f"{mdl_obj[0]} Prediction startet at: {time_start.strftime('%Y-%m-%d, %H:%M:%S')}")
        _pred = model.predict(X_validation_total, verbose=1)
        predicted_gait_phases.append(_pred)
        predicted_gait_phases[mdl_index] = np.vstack((np.zeros((LSTM_window_left, 1)), predicted_gait_phases[mdl_index], np.zeros((LSTM_window_right+1, 1))))
        y_validation_corrected.append(np.concatenate((np.zeros((LSTM_window_left, 1)), y_validation[mdl_index].reshape(-1, 1),  np.zeros((LSTM_window_right+1, 1)))))
        time_end = datetime.now()
        print(f"{mdl_obj[0]} prediction finished at: {time_end.strftime('%Y-%m-%d, %H:%M:%S')}")
        time_duration = time_end - time_start
        print(f"Duration: {time_duration}")
    
        ## Evaluation
        time_start = datetime.now()
        print(f"{mdl_obj[0]} evaluation startet at: {time_start.strftime('%Y-%m-%d, %H:%M:%S')}")
        _score = model.evaluate(x=X_validation_total, y=y_validation[mdl_index], verbose=1)
        scores.append(_score)
        time_end = datetime.now()
        print(f"Evaluation finished at: {time_end.strftime('%Y-%m-%d, %H:%M:%S')}")
        time_duration = time_end - time_start
        print(f"Duration: {time_duration}")
        print(f"Evaluation results for {mdl_obj[0]} Model: \tAccuracy: {scores[mdl_index][1]:0.4f}, Loss: {scores[mdl_index][0]:0.4f}")
        print("\n\n")


# In[] Visualising the results
_colors = "rbg"
x_timeline = list(range(len(validation_set_X[:, [0]])+1))

plt.clf()
plt.plot(validation_set_X[:, [0]], label = "gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label = "freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label = "rotMatMag", linewidth=1, linestyle="solid", marker=".")

for mdl_index, mdl_obj in enumerate(_models_addresses):
    plt.plot(y_validation_corrected[mdl_index], label = f"Real {mdl_obj[0]}", linewidth=1, color=_colors[mdl_index], linestyle="solid", marker="s")
    markerline, stemlines, baseline = plt.stem(x_timeline, predicted_gait_phases[mdl_index], _colors[mdl_index], markerfmt=f"{_colors[mdl_index]}o", label=f"Predicted {mdl_obj[0]}")
    plt.setp(stemlines, 'linewidth', 3), plt.setp(markerline, 'markersize', 3)

plt.title(f"Gait phase prediction")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Correcting the Results
predicted_gait_phases_corrected = []
for idx in range(len(_models_addresses)):
    peaks, properties = find_peaks(predicted_gait_phases[idx].flatten(), height=(.1, None), distance=100)
    predicted_gait_phases_corrected.append(np.zeros(len(predicted_gait_phases[idx])))

    for index, val in enumerate(predicted_gait_phases[idx]):
        if val >= 0.05:   # set the minimum probability to equal or greater than 0.05
            predicted_gait_phases_corrected[idx][index] = 1

# In[] Correct the predicted gait phases
print(f"No. of sample frames for validation dataset: {len(validation_set_X)}")
for mdl_index, mdl_obj in enumerate(_models_addresses):
    for ii, ival in enumerate(predicted_gait_phases_corrected[mdl_index]):
        if ival == 1:
            predicted_gait_phases_corrected[mdl_index][ii+1:ii+50] = 0

    _no_of_events = len(y_validation_corrected[mdl_index][y_validation_corrected[mdl_index] == 1])
    _no_of_phases_predicted = len(predicted_gait_phases_corrected[mdl_index][predicted_gait_phases_corrected[mdl_index] == 1])

    
    print(f"{mdl_obj[0]} labeled: {_no_of_events}")
    print(f"{mdl_obj[0]} predicted: {_no_of_phases_predicted}\n")

# In[] Plot the final results
plt.clf()
plt.plot(validation_set_X[:, [0]], label = "gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label = "freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label = "rotMatMag", linewidth=1, linestyle="solid", marker=".")

for mdl_index, mdl_obj in enumerate(_models_addresses):
    plt.plot(2*y_validation_corrected[mdl_index], label = f"Real {mdl_obj[0]}", linewidth=1, color=_colors[mdl_index], linestyle="solid", marker="s")
    markerline, stemlines, baseline = plt.stem(x_timeline, predicted_gait_phases_corrected[mdl_index], _colors[mdl_index], markerfmt=f"{_colors[mdl_index]}o", label=f"Predicted {mdl_obj[0]} (corrected)")
    plt.setp(stemlines, 'linewidth', 3), plt.setp(markerline, 'markersize', 3)

plt.title("Gait phase prediction")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Finish
