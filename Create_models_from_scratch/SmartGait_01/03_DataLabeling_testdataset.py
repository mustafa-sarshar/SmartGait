"""
Data labeling:
   *** Step phase labeling
       - HeelStrike
       - ToeOff
       - MidSwing
"""
# In[] Import the libraries
import pickle
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

# In[] Initialize the primary variables
dataset_address = "datasets//all_rawCSV//02_dataset//dataset_readyForLabeling.pkl"
sensor_positions = ["Ankle_R", "Ankle_L"]       # Select a name for each one based on their wearing position

# In[] Load the dataset from folder 02_dataset
with open(dataset_address, "rb") as _data:
    dataset_subs = pickle.load(_data)

# In[] Label the events Peak methods
"""
Add the three label features for ToeOff, MidSwing and HeelStrike gait events manually.
You may use the find_peaks function from scipy library if you wish.
from scipy.signal import find_peaks

In general, you can establish an array of zeros based on the length of the signal,
and change the value of the frames to 1 when the gait event takes place.

Below you find an example of labeling and adding the label feature for MidSwing gait event
of the sensor Ankle_L for the sample number 0.
Here the MidSwing event takes place at frames 130, 225 and 410.
"""
sensor = "Ankle_R"
sensor_id = sensor_positions.index(sensor)
feature = "gyrMag_filter_gaussian"
new_feature = feature+"_MidSwing"
sample_index = 0

SIGNAL_LENGTH = len(dataset_subs[sensor_id]["data"][sample_index][feature].values)
data_labeled = np.zeros(SIGNAL_LENGTH)
data_labeled[130] = 1   # if the MidSwing gait event takes place at frame 130
data_labeled[225] = 1   # if the MidSwing gait event takes place at frame 225
data_labeled[410] = 1   # if the MidSwing gait event takes place at frame 410

# In[] Add the label feature to the dataset
dataset_subs[sensor_id]["data"][sample_index][new_feature] = data_labeled.reshape(-1, 1)

# In[] Repeat the labeling for the other sensor
sensor = "Ankle_L"
sensor_id = sensor_positions.index(sensor)
feature = "gyrMag_filter_gaussian"
new_feature = feature+"_MidSwing"
sample_index = 0

SIGNAL_LENGTH = len(dataset_subs[sensor_id]["data"][sample_index][feature].values)
data_labeled = np.zeros(SIGNAL_LENGTH)
data_labeled[230] = 1   # if the MidSwing gait event takes place at frame 230
data_labeled[325] = 1   # if the MidSwing gait event takes place at frame 325
data_labeled[510] = 1   # if the MidSwing gait event takes place at frame 510

# In[] Add the label feature to the dataset
dataset_subs[sensor_id]["data"][sample_index][new_feature] = data_labeled.reshape(-1, 1)

# In[] Extract the features for test dataset
_feature_label = "MidSwing"
test_features_list = list()
labeling_features_list = list()
test_features_list.append(["gyrMag_filter_gaussian", "freeAccMag_filter_gaussian"])
labeling_features_list.append([f"gyrMag_filter_gaussian_{_feature_label}"])
test_features_list.append(["gyrMag_filter_gaussian", "freeAccMag_filter_gaussian"])
labeling_features_list.append([f"gyrMag_filter_gaussian_{_feature_label}"])
features_X = list()
features_y = list()
_subdata_index = 0
_label_index = 3
for _dbi, _dbval in enumerate(dataset_subs):
    if dataset_subs[_dbi]["position"] in sensor_positions:
        print(f"Features from {dataset_subs[_dbi]['position']}_{_dbi}_{len(dataset_subs[_dbi]['data'])} added to the test dataset.")
        for _dfx, _dfval in enumerate(dataset_subs[_dbi]["data"]):
            _X1 = _dfval.loc[:, test_features_list[_subdata_index]].values
            _X2_1 = dataset_subs[0]["data"][_dfx].loc[:, ["Mat[1][1]_filter_gaussian"]].values
            _X2_2 = dataset_subs[1]["data"][_dfx].loc[:, ["Mat[1][1]_filter_gaussian"]].values
            _X2 = (_X2_1**2 + _X2_2**2)**0.5            
            _X = np.concatenate((_X1, _X2), axis=1)
            features_X.append(_X)
            _y = _dfval.loc[:, labeling_features_list[_subdata_index]].values
            features_y.append(_y)
        _subdata_index += 1    

# In[] Concatenate all the signals from both sensors and and scale the signal by MaxAbsScaler()
#      to generate one single scaled test dataset
test_dataset = np.concatenate((MaxAbsScaler().fit_transform(features_X[0]), features_y[0]), axis=1)
for i in range(1, len(features_X)):
    tmp = np.concatenate((MaxAbsScaler().fit_transform(features_X[i]), features_y[i]), axis=1)
    test_dataset = np.concatenate((test_dataset, tmp), axis=0)

data_labeled = test_dataset[:, _label_index]

# In[] save the sub dataset in folder test_dataset
dataset_address = f"datasets//test_dataset//test_dataset_{_feature_label}.pkl"
with open(dataset_address, "wb") as _output:
    pickle.dump(test_dataset, _output)
print(f"Test dataset for {_feature_label} is generated.")

# In[] Finish