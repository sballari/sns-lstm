"""Module that defines the SocialLSTM model."""
import tensorflow as tf
import trajectory_decoders


class SocialModel:
    """SocialModel defines the model of the Social LSTM paper."""

    def __init__(
        self,
        dataset,
        helper,
        position_estimate,
        loss_function,
        lstm_size=128,
        max_num_ped=100,
        trajectory_size=20,
        embedding_size=64,
        learning_rate=0.003,
        dropout=0.75,
    ):
        """Constructor of the SocialModel class.

        Args:
          dataset: A TrajectoriesDataset instance.
          helper: A coordinates helper function.
          position_estimate: A position_estimate function.
          loss_function: a loss funtcion.
          lstm_size: int. The number of units in the LSTM cell.
          max_num_ped: int. Maximum number of pedestrian in a single frame.
          trajectories_size: int. Length of the trajectory (obs_length +
            pred_len).
          embedding_size: int. Dimension of the output space of the embedding
            layers.
          learning_rate: float. Learning rate.
          dropout: float. Dropout probability.

        """
        # Create the tensor for input_data of shape
        # [max_num_ped, trajectory_size, 2]
        self.input_data = dataset.tensors[0]
        # Create the tensor for num_peds_frame
        self.num_peds_frame = dataset.tensors[1]

        # Store the parameters
        # In training phase the list contains the values to minimize. In
        # sampling phase it has the coordinates predicted
        self.new_coordinates = []
        # List that conatains the ground truth coordinates preprocessed
        self.coordinates_preprocessed = []
        # The predicted coordinates processed by the linear layer in sampling
        # phase
        new_coordinates_processed = None

        # Output size
        output_size = 5

        # Define the LSTM with dimension lstm_size
        with tf.name_scope("LSTM"):
            self.cell = tf.nn.rnn_cell.LSTMCell(lstm_size, name="Cell")

            # Define the states of the LSTMs. zero_state returns a tensor of
            # shape [max_num_ped, state_size]
            with tf.name_scope("States"):
                self.cell_states = self.cell.zero_state(max_num_ped, tf.float32)

        # Define the layer with ReLu used for processing the coordinates
        with tf.variable_scope("Coordinates"):
            self.coordinates_layer = tf.layers.Dense(
                embedding_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="Layer",
            )

        # Define the layer with ReLu used as output_layer for the decoder
        with tf.variable_scope("Position_Estimation"):
            self.output_layer = tf.layers.Dense(
                output_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="Layer",
            )

        # Define the SocialTrajectoryDecoder.
        # NOTE For now no pooling
        decoder = trajectory_decoders.SocialDecoder(
            self.cell,
            max_num_ped,
            helper,
            pooling_module=None,
            output_layer=self.output_layer,
        )

        # Processing the coordinates
        with tf.variable_scope("Coordinates_preprocessing"):
            for ped in range(max_num_ped):
                self.coordinates_preprocessed.append(
                    self.coordinates_layer(self.input_data[ped])
                )
            self.coordinates_preprocessed = tf.stack(self.coordinates_preprocessed)

        # Decode the coordinates
        for frame in range(trajectory_size - 1):
            # Initialize the decoder passing the real coordinates, the
            # coordinates that the model has predicted and the states of the
            # LSTMs. Which coordinates the model will use will be decided by the
            # helper function
            decoder.initialize(
                frame,
                self.coordinates_preprocessed[:, frame],
                new_coordinates_processed,
                cell_states=self.cell_states,
            )
            # compute_pass returns a tuple of two tensors. cell_output are the
            # output of the self.cell with shape [max_num_ped , output_size] and
            # cell_states are the states with shape
            # [max_num_ped, LSTMStateTuple()]
            cell_output, self.cell_states = decoder.step()

            # Compute the new coordinates
            new_coordinates, new_coordinates_processed = position_estimate(
                cell_output,
                self.input_data[:, frame + 1],
                output_size,
                self.coordinates_layer,
            )

            # Append new_coordinates
            self.new_coordinates.append(new_coordinates)

        # self.new_coordinates has shape [trajectory_size - 1, max_num_ped]
        self.new_coordinates = tf.stack(self.new_coordinates)

        with tf.variable_scope("Calculate_loss"):
            index = tf.constant(0, name="index")
            loss = tf.constant(0, tf.float32, name="loss")

            cond = lambda i, loss: tf.less(i, self.num_peds_frame)

            def body(i, loss):
                loss = tf.add(loss, loss_function(self.new_coordinates[:, i]))
                return tf.add(i, 1), loss

            _, self.loss = tf.while_loop(cond, body, [index, loss])

        # Define the RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.trainOp = optimizer.minimize(self.loss)