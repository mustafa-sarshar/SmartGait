"""
Train and Test the model
    - Initialize the train and test datasets
    - Vectorize the train and test datasets
    - Define and compile the model
    - Fit the model
"""
# In[] Import the libraries
import pickle, time, os, tensorflow, sys
from utils import preprocessing as preProc 
from utils import lstm_model

# In[] Initialize the variables
_feature_label = "MidSwing"
COLS_X = [0, 1, 2]
COLS_Y = [3]
LSTM_WINDOW_LEFT = 125
LSTM_WINDOW_RIGHT = 125

# In[] Import the dataset (Train set)
address = f"datasets//train_dataset//train_dataset_{_feature_label}.pkl"
with open(address, "rb") as _data:
    db_train = pickle.load(_data)

# In[] Import the dataset (Test set)
address = f"datasets//test_dataset//test_dataset_{_feature_label}.pkl"
with open(address, "rb") as _data:
    db_test = pickle.load(_data)
    
# In[] Initialize the train and test datasets
train_set_X = db_train[:, COLS_X]
train_set_y = db_train[:, COLS_Y]
test_set_X = db_test[:, COLS_X]
test_set_y = db_test[:, COLS_Y]

# In[] Train Set
higher_bound = len(train_set_X)-LSTM_WINDOW_RIGHT
X_train_total, y_train = preProc.dataset_vectorizing(train_set_X, train_set_y, LSTM_WINDOW_LEFT, LSTM_WINDOW_RIGHT, higher_bound)

# Clean up the memory from unnecessary variables
del db_train, train_set_X, train_set_y

# In[] Test Set
higher_bound = len(test_set_X)-LSTM_WINDOW_RIGHT
X_test_total, y_test = preProc.dataset_vectorizing(test_set_X, test_set_y, LSTM_WINDOW_LEFT, LSTM_WINDOW_RIGHT, higher_bound)

# Clean up the memory from unnecessary variables
del db_test, test_set_X, test_set_y
    
# In[] Build the LSTM model
## Set the strategy to train the Model
if tensorflow.config.list_physical_devices("GPU"):
    strategy = tensorflow.distribute.MirroredStrategy() # set to MirroredStrategy
    print("Strategy is set to MirroredStrategy")
else:  
    strategy = tensorflow.distribute.get_strategy() # set to the default strategy
    print("Strategy is set to DefaultDistributionStrategy")

with strategy.scope():  
    
    model = lstm_model.initialize_lstm_model(X_train_total)
    model = lstm_model.compile_lstm_model(model)
    monitor = lstm_model.set_monitor_lstm_model()
    
    model.summary()

    ## Fit the model
    time_start = time.localtime()
    print(f"Model for parameter '{_feature_label}' will be trained.")
    print(f"Model startet at: {time_start.tm_hour}:{time_start.tm_min}:{time_start.tm_sec}")
    model_history = lstm_model.fit_lstm_model(model, X_train_total, y_train, X_test_total, y_test, monitor, 1)    
    time_end = time.localtime()
    print(f"Model finished at: {time_start.tm_hour}:{time_start.tm_min}:{time_start.tm_sec}")


# In[] Create a new sub-directory based on the name of the labeld feature in the models directory
try:
    os.mkdir(path=f"models//{_feature_label}")
except ValueError:
    print(f"An exception occurred: {ValueError}")
except:
    print("Unexpected error:", sys.exc_info()[0])

# In[] Save the Model in the folder models
## Save the model in only one file
model.save(f"models//{_feature_label}//singleFileModel.h5")

## Save the history
with open(f"models//{_feature_label}//history.pkl", "wb") as output:
    pickle.dump(model_history.history, output)

## Save the weights only
model.save_weights(filepath=f"models//{_feature_label}//save_weights//weights") 

## Save the model completely
tensorflow.saved_model.save(model, f"models//{_feature_label}//mobilenet//")

# In[] Finished
