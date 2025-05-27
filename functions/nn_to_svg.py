import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


# --------------------
# Model Creation
# --------------------
def create_model(seq_length, n_features):
    """
    Creates an LSTM model for stock price prediction.

    Args:
        seq_length (int): Sequence length for LSTM model.
        n_features (int): Number of features in the input data.

    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(256, activation='tanh', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.4),
        LSTM(128, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(64, activation='tanh'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
    return model

example_seq_length = 30
example_n_features = 5

# Create an instance of your model
model = create_model(example_seq_length, example_n_features)

# Generate and save the model plot
output_filename = "lstm_model_diagram.png" # Changed to PNG for broader compatibility
plot_model(model, to_file=output_filename, show_shapes=True, show_layer_names=True, rankdir='TB') # TB for Top to Bottom

print(f"Neural network diagram saved to {output_filename}")
print("You can open this PNG file in any image viewer.")