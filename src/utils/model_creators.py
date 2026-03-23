from gossiplearning.config import Config

from keras.layers import Dropout
from tensorflow.keras import Model, Sequential, Input, layers
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow import reduce_mean


def create_LSTM(config: Config) -> Model:
    optz = Adam(learning_rate=0.001, epsilon=1e-6)

    input_timesteps = 4

    inputs = Input(shape=(input_timesteps, config.training.n_input_features))

    lstm_layers = Sequential(
        [
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
            ),
            LSTM(
                50,
                activation="tanh",
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )(inputs)

    outputs = [
        Dense(1, activation="relu", name=f"fn_{i}")(lstm_layers)
        for i in range(config.training.n_output_vars)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optz,
        loss={f"fn_{i}": "mse" for i in range(config.training.n_output_vars)},
        metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()],
    )

    return model


@register_keras_serializable()
def r2_score_training(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


@register_keras_serializable()
class JanossyPooling(layers.Layer):
    def call(self, inputs):
        return reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_janossy_rnn_model(config: Config) -> Model:
    """
    Builds a Janossy pooling model using GRU/LSTM and averaging over k permutations.

    Args:
        output_dim: Output dimension (e.g. 1 for regression)
        input_dim: Number of features per timestep (e.g. 8)
        rnn_type: 'gru' or 'lstm'
        rnn_units: Units in the recurrent layer
        num_permutations: Number of permutations per input sequence
    Returns:
        Compiled Keras model
    """
    reg_output_dim = config.training.reg_output_dim
    input_dim = config.training.input_dim
    rnn_type = 'gru'
    rnn_units = 80
    num_permutations = 6
    reg_output_activation = "linear"
    reg_loss_function = "mse"
    cls_output_activation = "sigmoid"
    cls_loss_function = "binary_crossentropy"

    # Input layer
    input_layer = Input(
        shape=(num_permutations, None, input_dim), name="input_layer"
    )

    # Mask padded zeros before processing
    x = layers.TimeDistributed(
        layers.Masking(mask_value=0.0),
        name="masking_layer"
    )(input_layer)

    # h(x): Embed each timestep feature
    x = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(64, activation='relu', name="dense_embedding")
        ),
        name="time_distributed_embedding"
    )(input_layer)  # -> (batch_size, num_permutations, sequence_length, 64)

    # f*: RNN across each permutation
    if rnn_type.lower() == 'gru':
        x = layers.TimeDistributed(
            layers.GRU(rnn_units, name="gru_rnn"),
            name="time_distributed_rnn"
        )(x)
    else:
        x = layers.TimeDistributed(
            layers.LSTM(rnn_units, name="lstm_rnn"),
            name="time_distributed_rnn"
        )(x)
    # -> (batch_size, num_permutations, rnn_units)

    x = JanossyPooling(name="janossy_pooling")(x) # -> (batch_size, rnn_units)

    # ρ: MLP for final prediction
    x = layers.Dense(128, activation='tanh', name="dense_final")(x)
    x = layers.Dropout(0.2, name="dropout_final")(x)
    
    # Output layers, loss and metrics definition
    # -- regression head
    outputs = [
        layers.Dense(
            reg_output_dim, 
            activation = reg_output_activation, 
            name = "fn_0"
        )(x)
    ]
    loss = {
        "fn_0": reg_loss_function
    }
    metrics = {
        "fn_0": [
            "mae", 
            "msle", 
            "mse", 
            "mape", 
            RootMeanSquaredError(), 
            r2_score_training
        ]
    }
    # -- classification head (if required)
    if config.training.n_output_vars > 1:
        outputs.append(
            layers.Dense(
                1, activation = cls_output_activation, 
                name = "fn_1"
            )(x)
        )
        loss["fn_1"] = cls_loss_function
        metrics["fn_1"] = ["accuracy"]

    # Build and compile model
    model = Model(
        inputs = input_layer, 
        outputs = outputs, 
        name = "janossy_multitask_model"
    )
    
    model.compile(
        optimizer = "adam",
        loss = loss,
        metrics = metrics,
    )

    return model

