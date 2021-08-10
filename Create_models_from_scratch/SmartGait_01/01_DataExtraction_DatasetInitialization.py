"""
Data extraction & dataset initialization
   - Data extraction & dataset initialization
   - Extract the sub dataset 
   - Extract necessary features
   - Save it as pickle
"""
# In[] Import the libraries
import os
import pickle
from utils import preprocessing as preProc

# In[] Initialize the primary variables
dataset_address  = "datasets//all_rawCSV"       # insert the address of the dataset including .csv files from MT Manager software
sensor_codes     = ["00B46A09", "00B46A02"]     # ID/code of the MTw sesnors 
sensor_positions = ["Ankle_R", "Ankle_L"]       # Select a name for each one based on their wearing position

_SKIP_ROWS       = 12                            # set this value based on the number of unnecessary rows in .csv files
_FILE_FORMAT     = ".csv"                        # specify the file format for searching the right files in the directory

# In[] Check all files exist in folder raw_data
rawdata_address = f"{dataset_address}//raw_data//"
file_list = os.listdir(rawdata_address)
file_list_filtered = [_s for _s in file_list if _FILE_FORMAT in _s]
print(f"\nNumber of Files: {len(file_list)}")
print(file_list_filtered, "\n")

# In[] Initialize the main dataset
dataset_all = preProc.dataset_initializing(address=rawdata_address, file_list=file_list, skiprows=_SKIP_ROWS)

# In[] Initialize the sub dataset
for _code, _position in zip(sensor_codes, sensor_positions):
    print(f"Sensor: {_code}-{_position}")
    dataset_sub = preProc.dataset_initializing_sub(data=dataset_all, sensorname=_code)
        
    ## Extract just necessary features
    _cols = ["FreeAcc_E", "FreeAcc_N", "FreeAcc_U", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mat[1][1]"]
    
    ## Note: The .csv file from MT Manager Software must include at least all of these features.
    for dfi, dfval in enumerate(dataset_sub):
        dataset_sub[dfi] = dfval.loc[:, _cols]
    
    ## save the dataset of sub sensors in folder 01_dataset
    address = f"{dataset_address}//01_dataset//{_position}-{_code}_01.pkl"
    with open(address, "wb") as output:
        pickle.dump(dataset_sub, output)
    print(f"File saved in {address}\n")

# In[] Finish