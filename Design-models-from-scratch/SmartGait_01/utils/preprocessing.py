"""
This module contains all required preprocessing methods/functions.
"""

def dataset_initializing(address, file_list, skiprows=12):
    """
    This function initializes a dataset based on given address.
    All .csv files will be loaded and based on the file names, they will be saved in a dictionary.
    The filenames may be as follows: 'participantID_sensorID_testID.csv'
    """
    import pandas as pd

    data_all = []
    for filename in file_list:
        xsens_dataset = pd.read_csv(address+filename, header=0, skiprows=skiprows)
        df = pd.DataFrame(xsens_dataset.loc[:, :], copy=True)
        participant_name = filename.split(".")[0].split("_")[0] # Extract the participant name/id from the filename
        sensor_name = filename.split(".")[0].split("_")[-1]     # Extract the sensor name/id from the filename
        test_code = filename.split(".")[0].split("_")[-2]       # Extract the test name/id from the filename
        df["sensor"] = sensor_name
        df["participant"] = participant_name
        df["testcode"] = test_code
        df["filename"] = filename
        dic = {"sensor":sensor_name, "participant":participant_name, "testcode":test_code, "filename":filename, "dataframe":df}
        data_all.append(dic)
    return data_all

def dataset_initializing_sub(data, sensorname):
    """
    This function creates a sub-dataset by extracting only the gait data from the main dataset based on given sensorname
    """
    data_sub = []
    for d in data:
        if d["sensor"] == sensorname:
            data_sub.append(d["dataframe"])
    return data_sub

def dataset_missingvalue_imputation(dataset, imputation_method, interpolation_method="cubic"):
    """
    This function performs missing value imputation based on given imputation method.
    """

    # if the imputation_method is set to interpolation, the missing values will be interpolated based on given interpolation_method
    if imputation_method == "interpolation":
        for df in dataset:
            for col in df:
                df[col].interpolate(method=interpolation_method, inplace=True)

def _dataset_outliers_removing(data, threshold=1):
    """
    This function is a sub-function of dataset_outliers_removing.
    It removes any value above the given threshold, and converts it to 0.
    """
    data = data.flatten()
    nOutliers = 0
    outliers = []
    for di, dval in enumerate(data):
        if dval>threshold:
            data[di] = 0
            nOutliers += 1
            outliers.append(f"i: {di}, val: {dval}")
    return data, outliers

def dataset_outliers_removing(dataset, cols, threshold=1):
    """
    This function removes the outliers for all given columns in the dataset based on given threshold.
    """
    import numpy as np

    for dfi, dfval in enumerate(dataset):
        for col in dfval:
            if col in cols:
                sigOut, outliers = _dataset_outliers_removing(
                    data=dfval[col].values,
                    threshold=threshold
                )
                print(f"Outliers solved for Column: {col} in DF_No.: {dfi}, No. of Outliers: {len(outliers)}, in {outliers}")
                dfval[col] = np.array(sigOut).reshape(-1, 1)

def dataset_signal_filtering(dataset, cols=[]):
    """
    This function filters the signal via Gaussian method
    """
    import numpy as np

    # This function convert the FWHM to Sigma
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    from utils.signal_processing import signal_filtering_gaussian
    for dfi, dfval in enumerate(dataset):
        for col in dfval:
            if col in cols:     
                sigOut = signal_filtering_gaussian(
                    data=dfval[col].values,
                    sigma=fwhm2sigma(9),
                    mode='nearest'
                )
                print(f"Signal filtered by Gaussian method for Column: {col} in DF_No.: {dfi}")
                dfval[col+"_filter_gaussian"] = np.array(sigOut).reshape(-1, 1)

def dataset_add_new_features(dataset, filter_name):
    """
    This function will calculate and add new features to the given dataset based on given filtername.
    These new features are:
        1) freeAccMag: free acceleration magnitude
        2) freeAccMagF: free acceleration magnitude (filtered)
        3) gyrMag: angular velocity magnitude
        4) gyrMagF: angular velocity magnitude (filtered)
    """

    import numpy as np

    for df in dataset:

        _freeAcc_E = df["FreeAcc_E"].values
        _freeAcc_N = df["FreeAcc_N"].values
        _freeAcc_U = df["FreeAcc_U"].values
        _freeAcc_EF = df["FreeAcc_E_"+filter_name].values
        _freeAcc_NF = df["FreeAcc_N_"+filter_name].values
        _freeAcc_UF = df["FreeAcc_U_"+filter_name].values
        freeAccMag = (_freeAcc_E**2 + _freeAcc_N**2 + _freeAcc_U**2)**0.5
        freeAccMagF = (_freeAcc_EF**2 + _freeAcc_NF**2 + _freeAcc_UF**2)**0.5

        _gyrX = df["Gyr_X"].values
        _gyrY = df["Gyr_Y"].values
        _gyrZ = df["Gyr_Z"].values
        _gyrXF = df["Gyr_X_"+filter_name].values
        _gyrYF = df["Gyr_Y_"+filter_name].values
        _gyrZF = df["Gyr_Z_"+filter_name].values
        gyrMag = (_gyrX**2 + _gyrY**2 + _gyrZ**2)**0.5
        gyrMagF = (_gyrXF**2 + _gyrYF**2 + _gyrZF**2)**0.5

        df["freeAccMag"] = np.array(freeAccMag).reshape(-1, 1)
        df["gyrMag"] = np.array(gyrMag).reshape(-1, 1)
        df["freeAccMag_"+filter_name] = np.array(freeAccMagF).reshape(-1, 1)
        df["gyrMag_"+filter_name] = np.array(gyrMagF).reshape(-1, 1)

def dataset_vectorizing(data_set_X, data_set_y, LSTM_window_left, LSTM_window_right, higher_bound):
    """
    This function vectorizes the dataset and returns a numpy array (Tensor) for training or testing purposes.
    """
    import numpy as np

    X_data0 = []
    X_data1 = []
    X_data2 = []
    y_data = []

    for i in range(LSTM_window_left, higher_bound):
        X_data0.append(data_set_X[i-LSTM_window_left:i+LSTM_window_right+1, [0]].reshape(-1, 1))
        X_data1.append(data_set_X[i-LSTM_window_left:i+LSTM_window_right+1, [1]].reshape(-1, 1))
        X_data2.append(data_set_X[i-LSTM_window_left:i+LSTM_window_right+1, [2]].reshape(-1, 1))
        y_data.append(data_set_y[i, [0]])

    # Convert them to array and reshape them if necessary
    X_data0 = np.array(X_data0)
    X_data0 = np.reshape(X_data0, (X_data0.shape[0], X_data0.shape[1], 1))
    X_data1 = np.array(X_data1)
    X_data1 = np.reshape(X_data1, (X_data1.shape[0], X_data1.shape[1], 1))
    X_data2 = np.array(X_data2)
    X_data2 = np.reshape(X_data2, (X_data2.shape[0], X_data2.shape[1], 1))
    y_data = np.array(y_data)

    # Concatenate them to create on signle data set
    X_data_total = np.concatenate((X_data0, X_data1, X_data2), axis=2).astype('float16')

    return X_data_total, y_data
