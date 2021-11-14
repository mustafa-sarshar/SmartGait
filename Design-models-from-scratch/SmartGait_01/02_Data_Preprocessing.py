"""
Data preprocessing:
   - Missing value imputation
   - Remove the outliers
   - Smooth and filter the data
   - Add new features to the dataset
   - Cut the dataset and reduce its size if necessary
"""
# In[] Import the libraries
import os
import pickle
import pandas as pd
from utils import preprocessing as preProc

# In[] Initialize the primary variables
dataset_address  = "datasets//all_rawCSV"       # insert the address of the dataset including .csv files from MT Manager software
sensor_positions = ["Ankle_R", "Ankle_L"]       # Select a name for each one based on their wearing position

# In[] Read the files from folder 01_dataset
dataset_subs = []
for _position in (sensor_positions):
    file_list = os.listdir(f"{dataset_address}//01_dataset//")
    _filter_criteria = _position
    file_list_filtered = [s for s in file_list if _filter_criteria in s]    
    print(file_list_filtered)

    ## Load the dataset from folder 01_dataset
    dataset_sub = list()
    for addi, addval in enumerate(file_list_filtered):
        address = f"{dataset_address}//01_dataset//{addval}"
        with open(address, "rb") as data:
            temp = pickle.load(data)
        for xi, xval in enumerate(temp):
            dataset_sub.append(xval)

    ## Missing value imputation
    pd.options.mode.use_inf_as_na = True   # to consider inf and -inf to be “NA” in computations
    preProc.dataset_missingvalue_imputation(dataset=dataset_sub, imputation_method="interpolation", interpolation_method="pad")

    ## Remove outliers
    _col_list = ["FreeAcc_E", "FreeAcc_N", "FreeAcc_U"]
    preProc.dataset_outliers_removing(
            dataset=dataset_sub,
            cols=_col_list,
            threshold=160+1)

    _col_list = ["Gyr_X", "Gyr_Y", "Gyr_Z"]
    preProc.dataset_outliers_removing(
            dataset=dataset_sub,
            cols=_col_list,
            threshold=35+1)

    ## Smooth and Filter the necessary featrues
    _col_list = ["FreeAcc_E", "FreeAcc_N", "FreeAcc_U", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mat[1][1]"]
    preProc.dataset_signal_filtering(
            dataset=dataset_sub,
            cols=_col_list)

    ## Add new features to the dataset
    preProc.dataset_add_new_features(dataset=dataset_sub, filter_name="filter_gaussian")
    dataset_subs.append({"position":_position, "data": dataset_sub})
    print(f"Data from sensor '{_position}' added to dataset_subs.")
    print(f"The file saved to: {address}")

# In[] Cut the dataset
"""
Set cutoff points to cut the sample data and reduce its size if necessary.
Below an example of cutting the sample data '0' from frame 100 to 3000
"""
sample_index = 0
_cutoff_index_a = 100
_cutoff_index_b = 3000
for _namei, _nameval in enumerate(sensor_positions):
    dataset_subs[_namei]["data"][sample_index] = dataset_subs[_namei]["data"][sample_index][_cutoff_index_a:_cutoff_index_b+1]
    dataset_subs[_namei]["data"][sample_index].reset_index(drop=True, inplace=True)

# In[] save the sub dataset in folder 02_dataset
_dataset_subfolder = "all_rawCSV"
_filename = "dataset_readyForLabeling"
dataset_address = f"datasets//all_rawCSV//02_dataset//{_filename}.pkl"
with open(dataset_address, "wb") as output:
    pickle.dump(dataset_subs, output)

# In[] Finish