import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the input shape based on your time-series window length and number of features.
time_steps = 10    # Example sequence length
num_features = 5   # Adjust based on your feature set

# Time-series input for the LSTM.
input_seq = Input(shape=(time_steps, num_features), name="time_series_input")

# LSTM layer to capture sequential dependencies.
lstm_out = LSTM(64, return_sequences=False)(input_seq)
drop = Dropout(0.2)(lstm_out)

# Output for delay occurrence (binary classification).
delay_occurrence = Dense(1, activation='sigmoid', name="delay_occurrence")(drop)

# Output for delay duration (regression).
delay_duration = Dense(1, activation='linear', name="delay_duration")(drop)

# Output for location prediction (categorical, e.g., station index).
num_locations = 20  # Adjust this number based on your location data.
location_prediction = Dense(num_locations, activation='softmax', name="location_prediction")(drop)

# Build the model with multi-output.
model = Model(inputs=input_seq, outputs=[delay_occurrence, delay_duration, location_prediction])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'delay_occurrence': 'binary_crossentropy',
        'delay_duration': 'mse',
        'location_prediction': 'categorical_crossentropy'
    },
    metrics={
        'delay_occurrence': 'accuracy',
        'delay_duration': 'mse',
        'location_prediction': 'accuracy'
    }
)

model.summary()
