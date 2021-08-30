def dataset_initializing(
        address,
        file_list,
        skiprows=12):
    
    import pandas as pd

    data_all = []
    for fni, fnval in enumerate(file_list):
        xsens_dataset = pd.read_csv(address+fnval, header=0, skiprows=skiprows)
        df = pd.DataFrame(xsens_dataset.loc[:, :], copy=True)
        participant_name = fnval.split(".")[0].split("_")[0]
        sensor_name = fnval.split(".")[0].split("_")[-1]
        test_code = fnval.split(".")[0].split("_")[-2]
        df["sensor"] = sensor_name
        df["participant"] = participant_name
        df["testcode"] = test_code
        df["filename"] = fnval
        dic = {"sensor":sensor_name, "participant":participant_name, "testcode":test_code, "filename":fnval, "dataframe":df}
        data_all.append(dic)
    
    return data_all

def dataset_initializing_sub(
        data,
        sensorname):
     
    data_sub = []
    for dsi, dsval in enumerate(data):
        if dsval["sensor"] == sensorname:
            data_sub.append(dsval["dataframe"])
    
    return data_sub

def dataset_missingvalue_imputation(
        dataset,
        imputation_method,
        interpolation_method="cubic"):
    
    if imputation_method == "dropNA":
        return
    
    if imputation_method == "interpolation":
        for dfi, dfval in enumerate(dataset):
            for coli, colval in enumerate(dfval):
                dfval[colval].interpolate(method=interpolation_method, inplace=True)        
        return
    
    return

def _dataset_outliers_removing(
        data,
        threshold=1):
    
    data = data.flatten()
    nOutliers = 0
    outliers = []
    for di, dval in enumerate(data):
        if dval>threshold:
            data[di] = 0
            nOutliers += 1
            outliers.append(f"i: {di}, val: {dval}") 
    
    return data, outliers

def dataset_outliers_removing(
        dataset,
        cols,
        threshold=1):
    
    import numpy as np
    
    for dfi, dfval in enumerate(dataset):
        for coli, colval in enumerate(dfval):
            if colval in cols:
                sigOut, outliers = _dataset_outliers_removing(
                    data=dfval[colval].values,
                    threshold=threshold)
                print(f"Outliers solved for Column: {colval} in DF_No.: {dfi}, No. of Outliers: {len(outliers)}, in {outliers}")
                dfval[colval] = np.array(sigOut).reshape(-1 ,1)
    
    return

def dataset_signal_filtering(
        dataset,
        cols=[]):
    
    import numpy as np

    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))
                    
    
    from utils.signal_processing import signal_filtering_gaussian
    for dfi, dfval in enumerate(dataset):
        for coli, colval in enumerate(dfval):
            if colval in cols:     
                sigOut = signal_filtering_gaussian(
                    data=dfval[colval].values,
                    sigma=fwhm2sigma(9),
                    mode='nearest')
                print(f"Signal filtered by Gaussian method for Column: {colval} in DF_No.: {dfi}")
                dfval[colval+"_filter_gaussian"] = np.array(sigOut).reshape(-1 ,1)
    
    return

def dataset_add_new_features(
        dataset,
        filter_name):
    
    import numpy as np
    for dfi, dfval in enumerate(dataset):
        
        _freeAcc_E = dfval["FreeAcc_E"].values
        _freeAcc_N = dfval["FreeAcc_N"].values
        _freeAcc_U = dfval["FreeAcc_U"].values
        _freeAcc_EF = dfval["FreeAcc_E_"+filter_name].values
        _freeAcc_NF = dfval["FreeAcc_N_"+filter_name].values
        _freeAcc_UF = dfval["FreeAcc_U_"+filter_name].values
        freeAccMag = (_freeAcc_E**2 + _freeAcc_N**2 + _freeAcc_U**2)**0.5
        freeAccMagF = (_freeAcc_EF**2 + _freeAcc_NF**2 + _freeAcc_UF**2)**0.5
        
        _gyrX = dfval["Gyr_X"].values
        _gyrY = dfval["Gyr_Y"].values
        _gyrZ = dfval["Gyr_Z"].values
        _gyrXF = dfval["Gyr_X_"+filter_name].values
        _gyrYF = dfval["Gyr_Y_"+filter_name].values
        _gyrZF = dfval["Gyr_Z_"+filter_name].values
        gyrMag = (_gyrX**2 + _gyrY**2 + _gyrZ**2)**0.5
        gyrMagF = (_gyrXF**2 + _gyrYF**2 + _gyrZF**2)**0.5
        
        dfval["freeAccMag"] = np.array(freeAccMag).reshape(-1 ,1)
        dfval["gyrMag"] = np.array(gyrMag).reshape(-1 ,1)
        dfval["freeAccMag_"+filter_name] = np.array(freeAccMagF).reshape(-1 ,1)
        dfval["gyrMag_"+filter_name] = np.array(gyrMagF).reshape(-1 ,1)
            
    return

def dataset_vectorizing(
        data_set_X,
        data_set_y,
        LSTM_window_left,
        LSTM_window_right,
        higher_bound,
        ):
    
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