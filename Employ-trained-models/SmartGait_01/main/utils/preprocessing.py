def dataset_vectorizing(data_set_X, data_set_y, LSTM_window_left, LSTM_window_right, higher_bound):
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
