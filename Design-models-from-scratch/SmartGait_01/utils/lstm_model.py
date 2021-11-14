def initialize_lstm_model(
        X_train_total
    ):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import LSTM

    _NUM_OF_UNITS_INPUT_LAYER = 30
    _NUM_OF_UNITS_HIDDEN_LAYER_1 = 60
    _NUM_OF_UNITS_HIDDEN_LAYER_2 = 60
    _NUM_OF_UNITS_HIDDEN_LAYER_3 = 60
    _NUM_OF_UNITS_HIDDEN_LAYER_4 = 30
    _NUM_OF_UNITS_OUTPUT_LAYER = 1
    _DROPOUT_PERCENT = 0.2
    _INPUT_LAYER_ACTIVATION_METHOD = "tanh"
    _INPUT_LAYER_RECURRENT_ACTIVATION_METHOD = "sigmoid"
    _OUTPUT_LAYER_ACTIVATION_METHOD = "sigmoid"

    model = Sequential()

    # Add the Input layer: Layer 1
    model.add(LSTM(units=_NUM_OF_UNITS_INPUT_LAYER, activation=_INPUT_LAYER_ACTIVATION_METHOD, recurrent_activation=_INPUT_LAYER_RECURRENT_ACTIVATION_METHOD, return_sequences=True, input_shape=(X_train_total.shape[1], X_train_total.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    # Add the 1st Hidden Layer: Layer 2
    model.add(LSTM(units=_NUM_OF_UNITS_HIDDEN_LAYER_1, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    # Add the 2nd Hidden Layer: Layer 3
    model.add(LSTM(units=_NUM_OF_UNITS_HIDDEN_LAYER_2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    # Add the 3rd Hidden Layer: Layer 4
    model.add(LSTM(units=_NUM_OF_UNITS_HIDDEN_LAYER_3, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    # Add the 4th Hidden Layer: Layer 5
    model.add(LSTM(units=_NUM_OF_UNITS_HIDDEN_LAYER_4, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    # Add the output layer: Layer 6
    model.add(Dense(units=_NUM_OF_UNITS_OUTPUT_LAYER, activation=_OUTPUT_LAYER_ACTIVATION_METHOD))

    return model

def compile_lstm_model(
        model,
        optimization_option="adam",
        loss_function="binary_crossentropy",
        metrics=["accuracy", "mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error"]
):
    from tensorflow.keras.optimizers import Adam

    if optimization_option == "adam": optimization_option = Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=optimization_option, loss=loss_function, metrics=metrics)

    return model

def set_monitor_lstm_model(
    verbose=1,
    monitor="loss",
    patience=20
):
    from tensorflow.keras.callbacks import EarlyStopping

    monitor = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        mode="auto",
        restore_best_weights=True
)
    return monitor

def fit_lstm_model(
        model,
        X_train_total,
        y_train,
        X_test_total,
        y_test,
        monitor,
        epochs=100
):
    history = model.fit(
        X_train_total, y_train,
        validation_data=(X_test_total, y_test),
        callbacks=[monitor],
        epochs=epochs
)
    return history