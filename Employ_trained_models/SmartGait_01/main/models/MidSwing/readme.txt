Information about the datasets and the model:

Trained on: 197306 frames
Tested on:76012

Model structure specifications:
Epochs: 100
Patience: 50
Training took: 8:03:50.539725
No. of units (input layer): 30
No. of units (hidden layers):
    1) 60
    2) 60
    3) 60
    4) 30
No. of units (output layer): 1
Dropout percent: 0.2
Input layer activation method: tanh
Hidden layer activation method: tanh
Ourput layer activation method: sigmoid
Metrics: ['accuracy', <function mean_absolute_error at 0x000001F82115EF78>, <function mean_squared_error at 0x000001F82113F318>, <function poisson at 0x000001F821166708>, <function mean_squared_logarithmic_error at 0x000001F8211660D8>]
Optimization method: Adam
Optimization Hypers:
    1) Learning Rate: (<tf.Variable 'Adam/learning_rate:0' shape=() dtype=float32, numpy=0.001>,)
    2) Decay: (<tf.Variable 'Adam/decay:0' shape=() dtype=float32, numpy=1e-05>,)
    3) Beta 1: (<tf.Variable 'Adam/beta_1:0' shape=() dtype=float32, numpy=0.9>,)
    4) Beta 2: (<tf.Variable 'Adam/beta_2:0' shape=() dtype=float32, numpy=0.999>,)
Loss function: binary_crossentropy
