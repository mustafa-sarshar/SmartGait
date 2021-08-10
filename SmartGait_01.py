# Import the libraries
import pickle, tensorflow
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
from main.data_SmartGait_01.utils import preprocessing as preProc

# In[] Initialize the primary variables
_gait_phase = "MidSwing"   # change the _gait_phase variable to either FootOff or FootContact to load and run dataset and model, for FootOff or FootContact, respectively.
_dataset_address = f"main//data_SmartGait_01//datasets//{_gait_phase}//example_data.pkl"
_model_address = f"main//data_SmartGait_01//models//{_gait_phase}//mobilenet"

# In[] Load the dataset
db_validation = []
with open(_dataset_address, "rb") as _data:
    db_validation = pickle.load(_data)

# In[] Initialize the constants
columns_X = [0, 1, 2]
columns_y = [3]
ncolumns_X = len(columns_X)
ncolumns_y = len(columns_y)
LSTM_window_left = 125
LSTM_window_right = 125
LSTM_windows_size = LSTM_window_left + LSTM_window_right + 1

# In[] Initializing the train set
validation_set_X = db_validation[:, columns_X]
validation_set_y = db_validation[:, columns_y]

# In[] Validation Set
higher_bound = len(validation_set_X)-LSTM_window_right
X_validation_total, y_validation = preProc.dataset_vectorizing(validation_set_X, validation_set_y, LSTM_window_left, LSTM_window_right, higher_bound)

# In[] Load the Model
if tensorflow.config.list_physical_devices("GPU"):
    strategy = tensorflow.distribute.MirroredStrategy() # set to MirroredStrategy
    print("Strategy is set to MirroredStrategy")
else:
    strategy = tensorflow.distribute.get_strategy() # set to the default strategy
    print("Strategy is set to DefaultDistributionStrategy")

with strategy.scope():
    model = models.load_model(_model_address)
    model.summary()

    ## Prediction
    time_start = datetime.now()
    print(f"Prediction startet at: {time_start.strftime('%Y-%m-%d, %H:%M:%S')}")
    predicted_peaks = model.predict(X_validation_total, verbose=1)
    predicted_peaks = np.vstack((np.zeros((LSTM_window_left, 1)), predicted_peaks, np.zeros((LSTM_window_right, 1))))
    y_validation_corrected = np.concatenate((np.zeros((LSTM_window_left, 1)), y_validation.reshape(-1, 1),  np.zeros((LSTM_window_right, 1))))
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

# In[] Visualising the results
plt.clf()
plt.plot(validation_set_X[:, [0]], label = "gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label = "freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label = "rotMatMag", linewidth=1, linestyle="solid", marker=".")

plt.plot(y_validation_corrected[:], label = "Real peaks", linewidth=3, color="magenta", linestyle="solid", marker="s")
plt.plot(predicted_peaks, label = "Predicted Peaks", linewidth=3, color="red", linestyle="solid", marker="s")

plt.title(f"Peaks Prediction for {_gait_phase}")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Correcting the Results
peaks, properties = find_peaks(predicted_peaks.flatten(),
                               height=(.1, None),
                               distance=100)
signal_peaks = np.zeros(len(predicted_peaks))

for index, val in enumerate(predicted_peaks):
    if val >= 0.05:   # set the minimum probability to equal or greater than 0.05
        signal_peaks[index] = 1

# In[] Correct the peakes
for ii, ival in enumerate(signal_peaks):
    if ival == 1:
        signal_peaks[ii+1:ii+50] = 0

_no_of_events = len(y_validation_corrected[y_validation_corrected == 1])
_no_of_events_predicted = len(signal_peaks[signal_peaks == 1])

print(f"No. of sample frames for validation set {len(validation_set_X)}")
print(f"{_gait_phase} labeled: {_no_of_events}")
print(f"{_gait_phase} predicted: {_no_of_events_predicted}")

# In[] Plot the final results
plt.clf()
plt.plot(validation_set_X[:, [0]], label = "gyrMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [1]], label = "freeAccMag", linewidth=1, linestyle="solid", marker=".")
plt.plot(validation_set_X[:, [2]], label = "rotMatMag", linewidth=1, linestyle="solid", marker=".")

plt.plot(2*y_validation_corrected[:], label = f"Real peaks {_gait_phase}", linewidth=3, color="magenta", linestyle="solid", marker="s")
plt.plot(signal_peaks, label = f"Predicted Peaks (corrected) {_gait_phase}", linewidth=3, color="red", linestyle="solid", marker="s")

plt.title(f"Peaks Prediction, Events: {_no_of_events_predicted} for {_gait_phase}")
plt.xlabel("Time Frame")
plt.ylabel("Peaks / Amplitude / Probability%")
plt.legend(loc="upper left")
plt.show()

# In[] Finish
